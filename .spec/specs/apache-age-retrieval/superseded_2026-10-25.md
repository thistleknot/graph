# Superseded AGE spec draft (2026-10-24)

Preserved when the current AGE scope and gates were corrected on 2026-10-25.

## Requirements

# Apache AGE Retrieval

## Current disposition — 2026-10-24

Entry layer: Behavioral + Structural — add a PostgreSQL graph representation and a selectable retrieval backend while preserving the existing Open WebUI Function and SQL traversal.

## Observed baseline

- `docker-compose.yml` runs `pgvector/pgvector:0.8.1-pg18-trixie`; no AGE extension is declared in `sql/001_schema.sql`.
- `/agent` ranks dense and sparse vectors in PostgreSQL, expands similarity edges through `sect_edge_sym`, and bridges sections through normalized entity mention tables.
- `src/owui_filter.py` is a standalone Open WebUI Function that calls `/agent` and renders its report; it does not query JSONB directly.
- `sql/001_schema.sql` stores open-ended node/edge properties in JSONB; traversal keys remain typed, indexed columns.
- Apache AGE provides Cypher graph querying inside PostgreSQL and supports PostgreSQL 18 in the current official release line. The existing pgvector dependency therefore remains required.

## Requirements

- The graph database image SHALL include PostgreSQL 18, pgvector 0.8.1, and Apache AGE 1.7.0.
- The SQL traversal backend SHALL remain the default unless a request explicitly selects AGE.
- When a section build is persisted, the graph store SHALL publish AGE Section vertices and SIMILAR_TO edges for that build in the same transaction as the relational snapshot.
- When retrieval selects the AGE backend, similarity expansion SHALL use parameterized Cypher over the matching build graph.
- When an existing relational section build is backfilled, the backfill SHALL create a matching AGE graph without changing the live build pointer.
- When retrieval uses AGE, dense and sparse candidate ranking SHALL continue to use the current exact pgvector scans and BM25 store.
- When retrieval uses the existing SQL backend, the current SQL traversal and response behavior SHALL remain available.
- The existing `src/owui_filter.py` SHALL remain unchanged as the reference Function.
- The AGE Open WebUI Function SHALL select AGE traversal through the API request contract and preserve the current report rendering and failure behavior.
- If the requested AGE graph is missing, incomplete, or mismatched to the live section build, then the AGE retrieval request SHALL fail explicitly without falling back to SQL traversal.
- When a new graph build fails before publication, then the previous live relational build and its AGE graph SHALL remain queryable.

## Acceptance

- Given three varied fixtures including ties and high-degree sections, When SQL and AGE traversals run over the same build, Then returned neighbors, scores, ordering, and caps SHALL match.
- Given the AGE backend and an unavailable graph, When `/agent` runs, Then the API SHALL return an explicit error and SHALL NOT silently use SQL traversal.
- Given the legacy and AGE Open WebUI Functions, When either is imported, Then both SHALL pass standalone syntax and request-contract checks and retain the existing renderer output.
- Given the custom image, When PostgreSQL starts, Then `vector` and `age` SHALL be available and Cypher SHALL create, match, and traverse vertices and edges.

## CLOSED

- AGE does not replace pgvector; vector ranking remains in exact PostgreSQL vector operators.
- JSONB payloads and Dunning term records remain where they serve non-traversal metadata. AGE replaces graph traversal representation, not every JSON-shaped column.
- The previous Open WebUI Function is not renamed or overwritten; it remains the default SQL-path reference.

Preconditions: PostgreSQL 18 and pgvector storage remain compatible with the existing named volume. AGE functions are loaded and `ag_catalog` is available on each connection that issues Cypher.

Failure modes: image build incompatibility, unavailable extension files, incomplete graph synchronization, Cypher/operator mismatch, or divergent traversal parity.



## Design

# Apache AGE Retrieval Design

Behavioral layer: mirror build-scoped similarity relationships into AGE, then choose Cypher at the retrieval boundary.

## Pipeline

`PostgreSQL image (pgvector + AGE) -> transactional build graph -> SQL or AGE traversal -> unchanged vector ranking and answer synthesis -> legacy or AGE Open WebUI Function`

## Structure

- `Dockerfile.pg-age` extends the PostgreSQL 18 pgvector image with a pinned AGE release.
- `docker-compose.yml` builds that image while preserving the current `graphdata` volume and PostgreSQL settings.
- `sql/002_age.sql` installs AGE on newly initialized databases; runtime AGE initialization installs it on an existing database.
- `age_graph.py` owns AGE initialization, graph snapshots, completeness checks, and similarity traversal; `section_store.py` invokes it before publishing new builds and exposes `--build-age-graph` for backfill.
- Each AGE graph is named `arxiv_section_<build_id>` and contains `Section` vertices keyed by build and ordinal, `SIMILAR_TO` relationships with the existing edge scores, and a `GraphBuild` completeness marker.
- `SectionGraph` selects the traversal backend; `sql` remains the default, while `age` uses AGE-backed similarity traversal.
- `AgentRequest.graph_backend` is constrained to `sql` or `age`; API output reports the backend used.
- `src/owui_filter_age.py` is a standalone AGE variant of the current Function and sets `graph_backend=age`.
- `src/owui_filter.py` remains unchanged and continues to use SQL traversal.

## Behavioral contract

1. Build persistence creates the AGE graph, Section vertices, and SIMILAR_TO edges before the build becomes live; any failure rolls back the new build.
2. The backfill command builds the current live AGE graph from relational section/edge rows without changing the live pointer.
3. Cypher receives build IDs, ordinals, exclusions, limits, and caps as AGE parameters; it does not interpolate user-controlled query text.
4. AGE traversal returns the same stable order and score shape as `neighbours`.
5. Existing SQL traversal remains available for parity checks and rollback; AGE errors are visible and do not trigger SQL fallback.
6. The API keeps vector retrieval, Dunning reweighting, evidence composition, model calls, and response rendering unchanged.

## Validation order

1. Build the custom image and verify both extensions in a disposable PostgreSQL container.
2. Run Cypher create/match/traversal smoke checks and graph property conversions.
3. Compare SQL and AGE similarity traversal over at least three varied small fixtures.
4. Verify API backend selection, missing-graph failure, and report backend metadata.
5. Verify both standalone Open WebUI Function files and their request payloads.
6. Only after those gates pass, synchronize the live graph and exercise the AGE Function against the hosted API.

Each synthetic rung is bounded to 15 minutes; live corpus graph synchronization is a production migration, not a discovery test.
