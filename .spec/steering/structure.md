---
inclusion: always
---

# Structure

## Layout

```
chunkgraph.py        build: CHUNK -> ... -> COMMUNITY -> SERVE (the pipeline)
salient_grams.py     vocabulary selection for compact trigram indexes
pg_store.py          persist a fitted ChunkGraph (jsonb + pgvector)
ingest_brown.py      corpus driver (NLTK Brown)
graph3d.py           3D export           render_graph.py   2D render
sql/                 MIGRATIONS ONLY -- runs once, alphabetically, at init
cookbook/            query snippets with psql placeholders; never auto-run
tests/               test_{ingest_brown,pg_store,render_graph,salient_grams}.py
                     test_schema.sql (DDL-level assertions)
docker-compose.yml   graphdb service, host :5433
```

Generated artifacts (`graph3d.html`, `graph_brown*.png`, `suite*.png`,
`irl_tank.html`) are currently tracked. They are outputs, not sources — untrack
once confirmed reproducible.

## Naming & documentation convention

**Module docstrings carry the spec.** `chunkgraph.py` and `salient_grams.py`
open with a staged pipeline table plus numbered **EARS guards** (`R1`…`R10`) in
WHEN/IF/WHERE form. `pg_store.py` states Require/Guarantee.

Preserve this. When adding behaviour, add or amend an `R`-guard in the module
docstring and reference it from the test that pins it. `.spec/` holds
cross-cutting design and rationale; the `R`-guards stay where they are, next to
the code they bind.

## Architectural decisions

**Bitemporal, supersede-never-delete.** Edges carry `valid_from`/`valid_to`;
re-ingesting a label supersedes the prior live run. History is queryable.
Indexes are partial on `WHERE valid_to IS NULL` for the live set.

**Dual-space provenance.** Every edge records `{sparse,dense,both}`. Provenance
is a first-class column (`edge_prov` index), not metadata — it is how a result
is explained and how a space's contribution is audited.

**Significance ranks, backbone connects** (R7). A k-sigma cut alone leaves
isolates, so per-node KNN neighbours are unioned in regardless of significance.

**`community.cid` is run-local.** `PRIMARY KEY (run_id, cid)` — a new ingest
mints entirely new communities. Any human-supplied community label therefore
needs explicit carry-forward across runs (member-set overlap), or SME effort is
destroyed on every ingest. This is the central constraint on the labelling work.

**Determinism boundary.** Membership is computed; labels are text.
- Louvain partition → deterministic, batch, at ingest, fixed seed.
- `keywords` (tf\*idf) → corpus-derived, the SME-checkable middle layer.
- Any model-authored name → a *draft* over a frozen cluster, never a join key,
  never indistinguishable from a confirmed one.

Subgraph Louvain is **not** a restriction of global Louvain — partitioning an
induced neighbourhood yields community ids with no stable mapping to the global
run. This is why partitioning is never done at query time.
