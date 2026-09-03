<!-- Spec: diagnostic gold set for retrieval outcomes · Task: playbook.md LATER (sqrt-anchor gate) · basis: operator 2026-09-02 "a threshold based on an aggregate needs to be tuned to a diagnostic expected set of outcomes" -->

# Diagnostic prompt set — expected outcomes written BEFORE running

Rules: expectations are stated per prompt, in advance, as deterministic
assertions over the walk (source mix, anchor identity, edge provenance) — no
LLM judges anything. A threshold (anchor keyness, sqrt allocation) may only be
tuned to move FAILING rows to passing without breaking passing rows. ~20 items
per the standing eval-sizing rule; items are power, pools are cost.

Measures per walk: `mix` = source counts over walked chunks; `anchors` =
sources/doc_ids of R1-validated anchors; `xprov` = provenance of walked
edges whose endpoints differ in source. All from `sampler.ef_evidence` +
node/edge reads.

## A — event recall (wiki-targeted)

| id | prompt | expected |
|---|---|---|
| A1 | how did aircraft carriers decide the battle of midway | wiki ≥ 70% of mix; wiki/28410 among anchors |
| A2 | what happened at the battle of the coral sea | wiki dominant; wiki/16278 among anchors |
| A3 | how does wikipedia describe hurricane damage | wiki dominant; storm community (storm/hurricane keywords) touched |
| A4 | history of the boat race between oxford and cambridge | wiki dominant; boat_race community touched |
| A5 | how did the battleship yamato sink | wiki dominant; ≥1 anchor doc whose title parses to a Yamato-related article |

## B — register-explicit (quotes-targeted)

| id | prompt | expected |
|---|---|---|
| B1 | a quote about courage | quotes ≥ 50% of mix |
| B2 | a quote about love and loss | quotes ≥ 50% of mix |
| B3 | a quote about books and reading | quotes ≥ 50% of mix |
| B4 | a quote about being broken | quotes ≥ 50% of mix |
| B5 | a quote about dreams and hope | quotes ≥ 50% of mix |

## C — period civic prose (brown-targeted)

| id | prompt | expected |
|---|---|---|
| C1 | jury trial grand jury investigation | brown ≥ 20% of mix (measured 2026-09-02: 26/86 — pins the status quo) |
| C2 | school children teacher education | brown present ≥ 10%; mixed is acceptable |
| C3 | church congregation sunday sermon | brown ≥ 20% of mix |
| C4 | city council tax revenue budget hearing | brown ≥ 20% of mix |

## D — cross-register bridges

| id | prompt | expected |
|---|---|---|
| D1 | once people are broken they cannot be fixed | ≥2 sources in mix; ≥1 walked cross-source edge with provenance dense or both |
| D2 | what the average reader is always on the lookout for | brown among anchor sources; ≥2 sources in mix |
| D3 | how a city rebuilds after a disaster | wiki+brown both ≥ 10% of mix |

## E — ambiguous-word controls (the courage class)

Expected outcomes here honor the CORPUS, not the asker's unstated intent —
that is this set's reason to exist.

| id | prompt | expected |
|---|---|---|
| E1 | what courage means in a losing battle | wiki-dominant is CORRECT (battle df=1699 spans games+war); assert quotes ≥ 2 chunks appear at all (floor, not share) |
| E2 | the boss battle at the end of the game | game community dominant (mario/zelda terms in walk vocabulary); war articles < 20% |
| E3 | the senator fought a losing battle over the tax bill | brown+political wiki ≥ 50% combined; game community < 10% — discriminates whether context words steer anchor mass away from the majority sense |

## Calibration run (2026-09-02, run bfa594df mixed-full-dual) — SET FROZEN

Measured at walker-default walk unless marked ef24 (= ef=24, T=0,
bridge_pairs=0, T9's tight setting). 13/20 passed as drafted; calibration
verdicts below. After this freeze, a failing row is a finding, never a prompt
to edit.

| id | measured | verdict |
|---|---|---|
| A1 | wiki 76/77, wiki/28410 anchored | PASS |
| A2 | wiki 87/96 BUT anchors wiki/1076,3440,6388 — NOT the Coral Sea article | **KNOWN-FAIL (finding):** 'coral'+'sea' anchor mass drifts to marine articles; expectation stands |
| A3 | wiki 81/81, storm community | PASS |
| A4 | wiki 46/46, boat_race community | PASS |
| A5 | wiki 75/75 | PASS |
| B1–B5 | quotes 6–31% at ef24 on the DUAL run; 40–53% on the SPARSE mixed-full run, same prompts and knobs | **KNOWN-FAIL (finding), recalibrated:** target = quotes ≥ 40% of the ef24 walk — achieved by the sparse run, so attainable; dense expansion dilutes register-targeted retrieval 2–4x. THE tuning target. |
| C1 | brown 26/86 (30%) | PASS |
| C2 | brown 23/88, brown anchors | PASS |
| C3 | brown 26/88 | PASS |
| C4 | brown 7/83 (8%), brown anchor present | recalibrated (prediction error, Brown's tax/civic content thin): brown anchor present AND brown ≥ 5% → PASS |
| D1 | 3 sources; cross-edges dense 298 / both 37 | PASS |
| D2 | 3 sources, brown anchor | PASS |
| D3 | brown 1/88 | recalibrated (prediction error, Brown lacks disaster-rebuild prose): wiki-dominant suffices → PASS |
| E1 | wiki 75/91, quotes 2 present | PASS |
| E2 | game 9 / war 0 | PASS |
| E3 | game 0 | PASS — context words DID steer anchor mass away from the majority sense |

Standing findings the frozen set now carries:
1. **B-class (the big one):** dense-space expansion dilutes register-explicit
   retrieval — quotes share drops from 40–53% (sparse run) to 6–31% (dual run)
   on identical prompts and knobs. Any fix (anchor keyness, source-aware
   expansion, sqrt allocation) is tuned to move B to ≥40% without breaking
   A/C/D/E.
2. **A2:** lexically ambiguous entity names ('coral sea') mis-anchor; the
   named-article anchor expectation stays as the discriminator.
