---
description: "Build-scoped Apache AGE traversal for arXiv section retrieval"
import: []
---

***definitions***

- :SectionBuild: is one published arXiv section snapshot with stable ordinal IDs and typed relational source rows.
- :AgeGraph: is the AGE representation of one SectionBuild's similarity edges and graph metadata.
- :SqlTraversal: is the existing relational neighbor expansion and default retrieval backend.
- :AgeTraversal: is Cypher neighbor expansion selected explicitly by the request.
- :LegacyFunction: is the existing Open WebUI Function that calls the API using its SQL default.
- :AgeFunction: is the separate Open WebUI Function that requests and requires AGE traversal.
- :PayloadJsonb: is non-topological data used by existing consumers for terms, lengths, sources, diagnostics, or payloads.

***implementation reqs***

- The PostgreSQL image must retain PostgreSQL 18 and pgvector 0.8.1 and add pinned Apache AGE 1.7.0.
- Each :AgeGraph: must be named from its positive SectionBuild ID and contain one Section vertex per ordinal plus each undirected SIMILAR_TO edge.
- Graph names and internal Cypher source must be SQL constants; Cypher values must use the third `cypher` argument as an AGE parameter map. Bulk graph snapshots must export relational rows with server-side `COPY` and use AGE's CSV loader rather than per-edge endpoint matching in Cypher.
- AGE setup must restore the caller transaction's `search_path` after each successful operation and roll back its setup on failure.
- Exact dense/sparse ranking, BM25, section text, communities, and response synthesis remain relational.
- Existing :PayloadJsonb: fields remain until active callers are proven obsolete; this feature moves section graph expansion into AGE.
- The API must report the backend actually used; explicit AGE selection must fail closed when the matching graph is unavailable.
- The source build and its new :AgeGraph: must commit together before the live pointer changes.
- Backfill must export relational edges with server-side COPY and must not materialize the live edge table in client memory.
- Staged CSV input must be written under the PostgreSQL server's `/tmp/age` directory, use positive stable IDs (`ordinal + 1`), preserve nullable edge scores, and be removed after the graph transaction ends.

***test reqs***

- A disposable container must expose vector 0.8.1 and AGE 1.7.0 and complete parameterized Cypher create, match, and traversal operations.
- On a disposable graph with 81,473 vertices, AGE's file loader must load a 10,000-edge batch within the 15-minute test bound; loaded counts and nullable scores must match the source rows. Traversal semantics are checked on three small varied graphs so scale loading and retrieval behavior remain separate gates.
- At least three varied graphs must compare SQL and AGE neighbors, ties, exclusions, scores, ordering, and per-seed caps.
- API tests must cover SQL default, explicit AGE, missing graph failure, and reported backend.
- Both standalone Functions must preserve their renderer contract; the :LegacyFunction: remains unchanged and the :AgeFunction: must require AGE confirmation.
- Live backfill is a production migration and must follow all synthetic gates plus a verified independent database backup.

***functional specs***

- When a :SectionBuild: is persisted, the system must create its :AgeGraph: transactionally before publishing the live pointer.
  - Given graph creation fails, then the prior live relational build and its graph remain queryable.
- When a request selects :SqlTraversal:, the API must use relational neighbors and report `sql`.
  - Given no backend is supplied, then `sql` remains the default.
- When a request selects :AgeTraversal:, the API must use the matching :AgeGraph: and report `age`.
  - Given the graph is missing or mismatched, then the request fails explicitly without SQL fallback.
- When an existing :SectionBuild: is backfilled, the system must build its :AgeGraph: without changing the live pointer.
- When either Open WebUI Function runs, it must call the same API and keep the selected model and response renderer contract.
  - Given the :AgeFunction: receives a non-AGE report, then it raises before model generation.
- When the legacy Function is used, it must remain an unchanged SQL-path reference; it does not itself query JSONB.
- When a field in :PayloadJsonb: has no graph-traversal role, the AGE migration must leave that field and its current consumers intact.
