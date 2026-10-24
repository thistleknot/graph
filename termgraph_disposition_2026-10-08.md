## 2026-10-08 Term graph over the book chunks: terms as nodes, three edge types (plan.md)

**Question (operator):** are graph nodes terms, vectors, or both, and can NPMI / DIRT find the node and edge classes?
**Decision under test:** nodes are canonical terms; a term's vector is an attribute used to mint its class; edges are one typed
multigraph (`co_mention` NPMI, `rel:<class>` DIRT, `similar_to` vector cosine). Probe: `.tmp/termgraph_probe.py`, reusing
`F:\Documents\dev\graph\entity_derive.py` and `relations.py`. Corpus: 119 chunks of the 16 skills the pool cites (not the 79 exact
ids: they do not map to file ordinals). Run with default guards, relaxed guards, and relaxed guards with code fences stripped.

| test | verdict | receipt line |
|---|---|---|
| P4 co-mention and vector similarity are different signals | **TESTED AND WON** (bar: Jaccard < 0.5) | `[P4] chance baseline ... Jaccard=0.008`, `[P4] terms compared=418 mean top5 Jaccard(co_mention, vector_cosine)=0.036`. Gini: co_mention `[impurity, splitting, candidate, 12, node]`, vector `[Gini impurity, Expected Gini, entropy, outcome, no]`: topical neighbours versus same-kind neighbours. |
| P1 inventory finds the domain's own terms | **TESTED AND LOST** (bar 12 of 16) | defaults: `skill names found 4/16`; relaxed guards: `found 10/16`; relaxed + no code: `found 10/16`. |
| P2 term classes reproduce across seeds, vector route (this is also the vector-only graph: kNN over term vectors, then Leiden) | **TESTED AND LOST** (bar: seed-ARI >= 0.8, more than one class) | Six corpora/guard settings. Five: `accepted_res=None typed=0/...`, best ARI 0.7683 (family, res 4.0) and 0.785 (all 1,330 chunks). Sixth (family, relaxed): `accepted_res=0.5 ari=1.0 typed=456/600` is ONE class of 456 terms, not a partition. |
| P2b term classes reproduce across seeds, co-mention route (Leiden over the NPMI edges, same 0.8 gate, connected nodes only) | **TESTED AND WON on one setting, LOST on two** | Family corpus (29 skill files, 66 chunks of ~1,800 characters, code fences stripped, default guards): `res=0.5 ARI 0.807, res=1.0 0.894, res=2.0 0.796, res=4.0 0.860`; accepted `res=4.0 classes=53`, about half singletons. Same method, family + relaxed guards: `0.188 / 0.526 / 0.516 / 0.643`, no partition. 16-skill subset, relaxed, no code: accepted at res 4.0 with 0.816 (65 classes). |
| P3 classes mean something (hand-written battery of 4 term groups, written by me: a second opinion only) | **TESTED AND LOST** at the bar (>= 0.7 purity on >= 3 groups); 3 of 4 groups beat a shuffled-label baseline | Family/default: impurity measures 0.80 (baseline 0.27), boosting family 0.60 (0.27), bagging family 1.00 (0.37), hyperparameters 0.20 (0.27). Subset/relaxed: 0.75, 0.50, 0.20, 0.40. Printed classes include coherent ones (`[splitting, criterion, impurity, Gini, Gini impurity, information gain, ...]`, `[bagging, averaging, bootstrap, aggregation, ...]`) and junk ones (`[they, must, draw, right]`). |
| P5 connector templates merge genitives where raw connectors do not | **NOT TESTED** | Template side is circular (the candidate step writes one `GEN` key). Raw side, tested: subset `9 keys -> 9 classes` (34 similarity edges); family `6 keys -> 6 classes` with `sim_edges=0`. Scale-limited: no distributional evidence either way. |
| P6 scale suffices | **TESTED AND LOST** for relations | `[P5] template: keys=2275 sim_edges=34`: relation classes are almost all singletons. Co-mention edges are plentiful (13,999) but the top ones are NPMI 1.00 at 3 shared chunks, rare identifiers such as `STACK_COLUMNS`. |

**Root cause of P1 (diagnosed from the live feature rows, not tuned away):** the closed-class cut (a first or last token present in
more than 30% of chunks and almost never capitalised) and the residual-IDF floor (0.5) are built for 2,521 mixed papers. On a
one-topic corpus every chunk is on topic, so "random forest" (first token "random" in 52 of 119 chunks, share 0.44) and "decision
tree" (residual IDF -0.048) score exactly 0. Widening to all 1,330 book chunks made it worse (4/16): formula variables fill the
inventory (`t0 t1`, `x1 n1`, `p1 p2`).

**Follow-up run, same day (both options from the first disposition were executed):** the whole `trees-ensembles` family as the
corpus (58 files, 165,428 characters, paragraph-packed chunks, a skill file counted as one "paper"), and typing by co-mention
communities. Result: the co-mention route reproduces on the family corpus with default guards and the vector route does not, but the
classes miss the purity bar and the reproducibility does not survive relaxing the guards. Two measurement defects found on the way:
the incumbent's "finest passing resolution" rule accepts a ONE-class partition at ARI 1.0, and ARI over a graph that includes
isolated nodes is inflated (they are singletons in every seed); the numbers above are connected-only for that reason.

**Not known:** whether the few-shot naming step adds anything (it never ran: the one accepted partition has junk classes beside
coherent ones, so naming it would label noise); whether a purity battery written by someone other than me agrees; whether more
text moves purity (the family corpus is 12k tokens after stripping code).

**Decision on the representation (what the evidence supports):** terms as nodes; `co_mention` and `similar_to` as separate edge
types (P4); classes minted from co-mention communities, not from context vectors (P2 vs P2b). Not supported: relation classes
(`rel:<class>`), which have 0 to 34 similarity edges at this scale; leave them out of the first graph.

**Singleton merge, executed (the cheaper option):** classes under 3 terms were merged into the class with the strongest summed NPMI,
or dropped when no edge. `[P3m] classes 53 -> 20, dropped terms 35`. Purity on the battery is unchanged (0.80 / 0.60 / 1.00 / 0.20),
because every battery term was already in a large class. The merge removes noise classes; it does not fix the hyperparameter group
(`max_depth` in a numeric-junk class, `n_estimators`, `learning_rate`, `min_samples_leaf`, `max_features` in four other classes).

**Scale run (8 families, executed after the hook pointed out the data was available):** trees-ensembles, classification,
regression, model-eval, data-prep, clustering-dimred, optimization, probability-stats from the master corpus; code fences stripped;
816 chunks, 396 papers, ~147k tokens (12x the single family); 1,500-term inventory.

| test | verdict | receipt line |
|---|---|---|
| Vector route at scale | **TESTED AND LOST** | `[P2] tried(res,ncl,ARI)=[(0.5,3,0.652),(1.0,18,0.480),(2.0,38,0.617),(4.0,69,0.729)] accepted_res=None typed=0/1500`; relaxed guards: best 0.747, `typed=0/1500`. 12x the text did not raise the best ARI (0.768 at 12k tokens, 0.729 and 0.747 at 147k). |
| Co-mention route at scale | **Gate passed, partition LOST on use** | default guards: `res=1.0 classes=108 connected-only 0.810`, accepted; relaxed guards: best 0.775, no partition. The partition is topic blobs: 9 classes of 48 to 197 terms (regression, statistics, classification, probability, clustering), the rest tiny. |
| P3 purity at scale | **NOT A VALID TEST at this partition** | `[P3] impurity / boosting / bagging / hyperparameters purity=1.00 labels=[2,2,2,2,2]` for all four groups: every battery term sits in the same 167-term class (`models, prediction, feature, classification, performance, ...`). Purity 1.00 is consistent with a blob; groups are not separated from each other. |
| P1 inventory | **TESTED AND LOST** (bar 12 of 16) | `skill names found 8/16` (default), `10/16` (relaxed). Heterogeneity helps the default guards (4 -> 7 -> 8 of 16 as the corpus broadens) but not to the bar. |
| Genitive rediscovery by distribution (the template-versus-raw test, raw side) | **TESTED AND LOST** at this scale | `raw GEN/connector: keys=7903 sim_edges=2 GEN keys=47 -> GEN classes=47` (relaxed: 103 keys -> 103 classes, 7 similarity edges). 47 distinct genitive connectors, none merged by shared entity pairs. The template side stays circular (one `GEN` key by construction). |
| Relation classes `rel:<class>` | **TESTED AND LOST** at this scale | `template: keys=7857 sim_edges=2` (relaxed: 14,679 templates, 7 edges): almost every template is its own class. |

**Hypothesis withdrawn:** the earlier note that the vector route failed for lack of text is contradicted by the 12x run. What
separated the two readings: if text were the limit, best ARI should rise with tokens; it did not (0.768 -> 0.729/0.747).

**Vector route diagnosed (8 families, one variable at a time from the default, plus a null control):**

```
[VD] default (ctx+spell, k_nn=15)        ARI by res 0.5/1/2/4 = 0.652 0.480 0.617 0.729   accepted=None
[VD] min_occ=5                           0.773 0.544 0.624 0.738                           accepted=None
[VD] k_nn=5                              0.303 0.486 0.649 0.718                           accepted=None
[VD] k_nn=40                             1.000 0.463 0.676 0.738                           accepted=0.5 (2 classes)
[VD] ctx only                            0.916 0.305 0.437 0.505                           accepted=0.5 (2 classes)
[VD] spelling only                       0.423 0.529 0.693 0.789                           accepted=None
[VD] NULL ctx only (columns shuffled)    1.000 0.115 0.266 0.405                           accepted=0.5 (1 class)
```

- **Verdict: the context vectors carry weak real structure and not enough to reproduce.** Real ctx-only beats the shuffled null at
  every resolution that gives more than 2 classes (0.305 vs 0.115, 0.437 vs 0.266, 0.505 vs 0.405), by 0.1 to 0.19 ARI, against a
  gate of 0.8. Rival reading "no structure at all" is rejected by that gap; rival "graph too sparse (k) or too few matches
  (min_occ)" is rejected because k=5/15/40 and min_occ=3/5 all land within 0.02 of each other at 4 resolution steps.
- The only seed-stable fine partition is spelling (0.789 at 54 classes, the closest to the gate): surface form, not type. The two
  "accepted" settings are 2-class partitions (a coarse split), which the one-class/two-class trap already rules out as types.
- **Reading:** context-PPMI over ~150k tokens does not recover node types reproducibly; neither adding text (12x) nor changing the
  graph or vector weights moved it.

**Genitive rediscovery, non-circular (no hand-written `GEN` rule; same candidate pairs, one template per connector):**

```
[P5b] 'of' vs 'of the': shared=4 cosine=0.037     'of' vs 'of a': shared=0 cosine=0.000
[P5b] "'s" (pairs reversed) vs 'of the': shared=4 cosine=0.041     "'s" (as written) vs 'of': shared=0 cosine=0.000
[P5b] control 'in' vs 'and': shared=2 cosine=0.021     control 'of' vs 'and': shared=0 cosine=0.000
```

- **TESTED AND LOST** (bar 0.3): the genitive connectors share 0 to 4 entity pairs out of 25 to 141 each, indistinguishable from the
  unrelated controls (0.000 to 0.021). Entity pairs almost never repeat at this size, so the pair-set cosine has nothing to compare.
  The earlier run behind the DIRT design used 918,707 sentences; this corpus has 28,318 candidate events.

**Representation decision, final for this campaign:** nodes = terms; edge types `co_mention` (NPMI) and `similar_to` (vector cosine)
are separate (P4 held at 8 families: Jaccard 0.062 vs chance 0.001); a topic layer from co-mention communities is available and
reproducible; node types and relation classes are NOT available from NPMI or DIRT at this scale and should not appear in the first
graph. The LLM few-shot step therefore has no statistical clusters to name for types; it could still name the 9 topic classes.

**Campaign status: every test in `plan.md` is now won or lost with a receipt line; none is NOT TESTED.** The template-key
version of the genitive test stays retired as circular; the non-circular version above replaced it.

**Promotion decision (final):** the builder is NOT promoted into `rlv2/`. Bars for promotion were P1 to P4 passing; P4 passed, P1
and P3 did not, and the node-type and relation-class outputs it would produce are not reproducible. What was promoted is the
knowledge: the representation decision above, the guard finding, and the three gate traps (skill note and memory).

**Not part of this campaign (new hypotheses, would need a new plan):** typing terms by an LLM few-shot over co-mention topic
classes plus vector neighbours (a supervised-smell route the operator has pushed back on before); a DIRT-scale corpus (~10^5 to
10^6 sentences) for relation classes; an independent labeller grading the 9 topic classes.

**Supported by this run (P4 only):** the vector signal and the co-mention signal rank different neighbours, so they stay separate
edge types (`similar_to`, `co_mention`) rather than one merged score. A vector-only graph was not run, so it is not ruled out.
Not settled: the class vocabulary, the relation classes.

