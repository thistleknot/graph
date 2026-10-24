# Plan: one typed term graph over the 79 source chunks, and whether it holds up

<!-- NO GOVERNING SPEC. Basis: operator 2026-10-08 ("terms for nodes and edges between terms, or both? ... few-shot over the
identified terms/entities from IT/DIRT/NPMI to determine node types and edges ... decide how we want the graph represented")
and "/goal draft this as plan.md and do it e2e until your done with your disposition(s) and promotion(s)".
Statuses live here for this one campaign; no ledger elsewhere. Scratch code in .tmp/. -->

Objective (operator's words): decide whether nodes are terms, vectors, or both, and let the corpus (NPMI / DIRT) tell us the node
classes and edge classes, with the LLM only few-shotting names onto what the statistics found.

## Decision being tested (design, stated before any run)

- Nodes = canonical terms. A term's vector (PPMI context + spelling) is an attribute of the node, used only to mint its class.
- Edges = one multigraph, each row tagged with its evidence type:
  `co_mention` (NPMI, undirected), `rel:<class>` (connector template classed by DIRT pair-set cosine, directed),
  `similar_to` (vector cosine, candidate only).
- The LLM never defines a class. It names clusters the statistics already made.

## Incumbent (no new statistics code)

`F:\Documents\dev\graph\entity_derive.py` (`derive`, `type_entities`, NPMI/G2/C-value, Leiden with seed-ARI >= 0.8 gate) and
`relations.py` (`anchor`, `candidates`, `template_pairsets`, `template_similarity`, `rel_class_partition`). No incumbent exists for
the three-edge-type multigraph or the few-shot naming step; those are the only new parts, and they live in `.tmp/` until promoted.

## Corpus

Chunks of the 16 skills the 748 pool questions cite (the three chunk files in `data/rl/`), de-duplicated by text. The exact
79 ids do not map cleanly to file ordinals, so the corpus is the skills' chunks, not the 79; the count is reported.

## Pre-registered tests (each names the rival reading and the observation that separates them)

| id | claim | rival | separator | pass bar |
|---|---|---|---|---|
| P1 | The inventory finds the domain's own terms | It finds generic vocabulary | Share of the 16 skill names (e.g. "random forest") present in the inventory | >= 12 of 16 |
| P2 | Term classes are real | Leiden partitions noise | Seed-ARI at the finest accepted resolution (incumbent gate) | ARI >= 0.8, else classes = -1 and the claim is NOT TESTED at this n |
| P3 | Classes mean something | Classes are spelling/frequency artefacts | Hand-labelled battery of >= 3 term groups (algorithms / impurity measures / hyperparameters): does each group stay together? | purity >= 0.7 on >= 3 groups; labelled by me, so a second opinion only |
| P4 | Co-mention and vector similarity are different signals | They are redundant | Overlap of each term's top-5 co-mention neighbours with its top-5 cosine neighbours | mean Jaccard < 0.5 supports three edge types; >= 0.5 says drop `similar_to` |
| P5 | Connector classes need templates, not raw connectors | Raw surface connectors already cluster | Co-class rate of `of` / `of the` / `'s` genitives over raw vs template keys | template co-class > raw (the failure recorded for the other corpus) |
| P6 | The scale is enough | 79 chunks (~36k tokens) is too small for floors of 3 | Count of edges surviving the G2 x NPMI x n>=3 floors | reported; < 50 edges = scale-limited, NOT TESTED for P5 |

## Steps

1. [DONE] Corpus and `derive` -> P1 lost (4/16 default, 10/16 relaxed); guard cause diagnosed from feature rows.
2. [DONE] `type_entities` -> P2 lost at this n (no seed-ARI >= 0.8), P3 NOT TESTED.
3. [DONE] Co-mention NPMI and vector kNN -> P4 won (0.036 vs chance 0.008); relations scale-limited (34 sim edges / 2,275 templates).
4. [DONE] Anchor, candidates, template vs raw -> P5 NOT TESTED (circular on templates).
5. [DONE] Class step, both options: (a) whole `trees-ensembles` family as corpus; (b) classes from co-mention communities. Co-mention
   route passes the 0.8 gate on the family corpus (connected-only 0.796-0.894, 53 classes) and misses the purity bar (2 of 4 groups
   >= 0.7); vector route (= the vector-only graph) fails everywhere. Graph JSON with both class columns:
   `.tmp/termgraph_out_family_nocode/graph.json`.
6. [SKIPPED] Few-shot naming: the one accepted partition has junk classes beside coherent ones; naming it would label noise.
7. [DONE] Disposition at the top of `dispositions.md` (updated after the follow-up). Promotions: measured notes in the
   `unsupervised-entity-relations` skill; memory `corpus-fraction-guards-zero-the-topic` (guards, one-class acceptance, isolated-node ARI).
   Builder NOT moved into `rlv2/`: P1 and P3 missed their pre-registered bars.

8. [DONE] Singleton merge: 53 -> 20 classes, 35 terms dropped, battery purity unchanged (0.80 / 0.60 / 1.00 / 0.20). Mechanism
   hypothesis recorded (co-mention = topic, vectors = kind); not separated from "too little text".

9. [DONE] Scale run, 8 families (816 chunks, ~147k tokens, code stripped): vector route LOST (best ARI 0.729/0.747, unchanged by 12x
   text, so the "too little text" reading is withdrawn); co-mention route passes the gate but yields topic blobs (9 classes of 48-197
   terms, every battery term in one class, so purity 1.00 is not a valid test); genitive rediscovery LOST (47 raw connectors -> 47
   classes, 2 similarity edges); relation classes LOST (7,857 templates, 2 edges); P1 8/16 default, 10/16 relaxed.

10. [DONE] Vector route diagnosed (one variable at a time plus a null control): real context vectors beat the shuffled null by only
   0.10-0.19 ARI at every resolution above 2 classes (0.305/0.437/0.505 vs 0.115/0.266/0.405); k_nn 5/15/40 and min_occ 3/5 within
   0.02 of each other; the only seed-stable fine partition is spelling (0.789), i.e. surface form, not type.
11. [DONE] Genitive test made non-circular (no hand-written GEN rule): 'of' vs 'of the' cosine 0.037 (bar 0.3), controls 0.000-0.021,
   so the genitive connectors are indistinguishable from unrelated ones at this size. Template-key version retired as circular.

Campaign closed: no test is left NOT TESTED; no next work item inside this campaign. Dispositions written; promotions made as
knowledge (skill and memory notes). The builder is NOT promoted into `rlv2/` (P1 and P3 missed their pre-registered bars).
Follow-ons that would need a new plan are listed at the end of the disposition.

## Verification

Every claim in the disposition quotes the probe's own output line. Seed-diff the class step (different seeds must be run, ARI reported).
No GPU, no network, no LLM call needed for steps 1-4.
