# Plan: how, for what is in hand (scratch; playbook.md is the ledger)

<!-- NO GOVERNING SPEC. Basis: operator 2026-10-07 ("plan.md extends playbook ... playbook feeds plan ... a level 2 cache for what you are working on"). RULE: statuses live ONLY in playbook.md; this file holds the how and the scratch for the items in hand, and is rewritten as they close. Never a second ledger. -->

Objective (operator's words): a topic map over sections; genre blocks not listed as topics; the 12 largest communities as cards on the plot; the markdown carrying the rest; Dunning terms beside every label; LLM summaries with the surviving terms in bold.

## In hand (ids are playbook.md lookup keys)

**Hybrid GraphRAG, global + subgraph (T172)**
- Built: `src/section_store.py` (sect_* tables + SQL: seeds, neighbours, summary ranking; persisted as build 32, `python -u src/section_store.py --tag xpa`) and `src/section_graphrag.py` (`SectionGraph(conn)`, `answer`, reads Postgres only); scratch drivers `.tmp/graphrag_try.py` (`EXCL=1` turns the genre exclusion on) and `.tmp/arms_probe.py` (what each seed arm returns). Plan: C:\Users\user\.claude\plans\jolly-soaring-moler.md.
- Genre exclusion is OFF by default (flag excluded c21 "Long-term memory for AI agents", 0.537 vs bar 0.533).
- Next: dense arm lands on generic-register sections for some questions (genre axes in the query/node vectors: try the arm B vectors, measured, not assumed); then the 20-question blind comparison (RRF alone / + subgraph / + global).

**LLM summaries (T164)**
- All 823 drafted (0 failed, $0.13). Still to do: `python .tmp/render_labelled.py xpa` to refresh the PNG/MD with them, and the survival test: grade a blind sample of 20 topic/genre/mixed from the summary alone, compare with surviving-term counts, report P and n.

**Genre flag, second signal (T168)**
- Pre-registered in playbook.md: mean non-letter share of body text; pick the threshold from the first 20 graded, score the 18 blind. Drop it if it does not separate.
- Also grade c2, c10, c14 (flagged genre, terms read as agent topics): possible false positives.

## Scratch

- Cards cannot bold (matplotlib); the markdown carries the bold.
- Live `arxiv_sect` map and Postgres: untouched; persisting is T165 and waits for the operator's go.
- Output pair now: `.tmp/sections_xpa_labelled_community_map.png` + `.tmp/sections_xpa_labelled_communities.md`.
- Hypotheses, not findings: low survival marks a non-topic block (n=15, c23 counter-case); the genre flag misses captions/code/lists.
