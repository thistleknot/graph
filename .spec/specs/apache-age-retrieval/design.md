# Apache AGE Retrieval Design

Status: implemented and live. PostgreSQL 18 / pgvector 0.8.1 / AGE 1.7.0 runs on the existing `graph_graphdata` volume; build 1163 has a verified AGE graph. A three-seed live SQL/AGE parity check passed. Its AGE query took about 30 seconds, so broader retrieval latency remains a measured operational risk rather than a correctness failure.

## Pipeline

`PG18 + pgvector image -> AGE extension -> server-side CSV staging -> AGE COPY-based graph loader -> API backend selection -> SQL or Cypher neighbors -> unchanged ranking/evidence/synthesis -> matching Open WebUI Function`

## Owned data and boundaries

- `sect_build`, `sect_node`, and `sect_edge` are the authoritative relational build and remain the source of truth.
- `arxiv_section_<build_id>` stores Section vertices keyed by build ID and ordinal, undirected SIMILAR_TO relationships, and a GraphBuild completion marker.
- AGE handles neighbor expansion only; exact pgvector scans, BM25, section metadata, community records, and answer generation keep their current owners.
- The `node.attrs` JSONB fields in the general ChunkGraph schema feed term-frequency and metadata consumers; they are not the active `/agent` section adjacency representation. Do not migrate them in this feature.
- `src/owui_filter.py` calls `/agent` and does not query JSONB. Keep it unchanged as the SQL path; `src/owui_filter_age.py` selects AGE and verifies the reported backend.

## Structure

- `Dockerfile.pg-age` installs AGE 1.7.0 into the PostgreSQL 18 pgvector 0.8.1 base.
- `sql/002_age.sql` handles empty-volume initialization; runtime initialization handles the existing database.
- `age_graph.py` owns graph creation, server-side CSV staging/loading, Cypher execution, graph marker validation, bounded backfill, and neighbors. The image provides a PostgreSQL-writable `/tmp/age` directory; server-side `COPY` reads authoritative relational rows there, and a fixed `rm` command removes only internally named staging files.
- `section_store.py` creates the graph in the same transaction as the source snapshot and exposes an explicit backfill command.
- `SectionGraph` routes only neighbors by `graph_backend`; `AgentRequest` defaults to `sql` and reports the selected backend.

## Behavioral decisions

- Build IDs are the graph identity and are converted only into a fixed-prefix identifier after positive-integer validation.
- The graph name and dollar-quoted Cypher source are internal SQL constants; Cypher data values use the third argument as an AGE parameter map. Snapshot data uses AGE's `load_labels_from_file` and `load_edges_from_file` COPY-based loader.
- AGE setup runs inside a savepoint, restores the caller's `search_path` on success, and rolls back its setup on failure.
- SQL remains the default and a parity/reference route; AGE errors never select SQL implicitly.
- Build and backfill export positive vertex IDs (`ordinal + 1`) and edge CSV rows from the authoritative relational rows with server-side `COPY`, then invoke AGE's file loader inside the source transaction. Remove staged files after use and publish the completion marker only after loaded counts match.
- Server-side `COPY` and AGE's loader keep the full edge table out of client memory; do not restore a whole-build `fetchall()` path or per-edge endpoint matching during snapshot creation.
- Runtime checks compare the graph marker with the live relational build. A full graph cardinality scan must not run on each retrieval; one-time backfill validation compares stored counts before use.

## Gate order

1. Build the pinned image and inspect extensions in a disposable container.
2. Exercise parameterized Cypher and property/index syntax there.
3. Run three small parity shapes and API/Function contract checks. **Passed**; 32 section-store tests and 78 API/Function tests passed on disposable PostgreSQL.
4. Verify bulk-loader scaling and nullable scores. **Passed**; 81,473 vertices plus 10,000 edges loaded with exact counts, null scores preserved, and staging files removed in 4.25 seconds. Per-edge Cypher matching exceeded three minutes for the same 10,000-edge shape and was stopped.
5. Save the complete independent database archive on H: and restore the AGE source tables. **Passed for feature scope**; archive is 39 files / 7.35 GB and build 1163 counts match. A full restore of unrelated tables remains unverified because index recreation exceeded the 15-minute test bound.
6. Recreate PostgreSQL on the existing volume, then backfill live build 1163 with the AGE bulk loader. **Passed**; image healthy, `/tmp/age` is postgres-owned, and the live pointer stayed on build 1163.
7. Verify live graph counts, SQL parity, and API/Function contracts. **Passed**; graph has 81,473 vertices, 1,613,828 relationships, one matching completion marker, and empty staging directory. Three live seeds returned identical SQL and AGE neighbors (10 each). API backend selection/error and both Function contracts passed. Hosted Open WebUI execution was not part of this source-code task.

Production backfill is complete. Full restore of unrelated tables remains unverified; the tested archive and focused section-table restore cover the AGE source data.
