"""| Stage | Mechanism | Tag | Ref |
|---|---|---|---|
| ENABLE | Load AGE and set transaction-local ag_catalog lookup. | [mandatory] | R1 |
| SNAPSHOT | Export relational rows with server-side COPY and load AGE labels before publishing the build. | [mandatory] | R2-R6, R11-R12 |
| BACKFILL | Stream source rows server-side into AGE's COPY-based graph loader. | [optional] | R6, R11-R12 |
| EXPAND | Use Cypher to rank bounded neighbors for explicitly selected AGE retrieval. | [optional] | R7-R9 |
| VALIDATE | Compare the completion marker with the owning relational build. | [contract] | R10 |

R1 The runtime image supplies AGE binaries and extension files; AGE setup restores the caller search_path on success and rolls back on failure.
R2 A graph name is derived only from a positive database build ID; its name and Cypher source are SQL constants, and Cypher values use the third argument as a bound agtype map.
R3 The graph snapshot and relational source build commit together before the source live flag changes.
R4 Loaded vertex and SIMILAR_TO edge counts must equal relational source counts; the marker is written last.
R5 AGE expands section neighbors only; exact vector ranking, BM25, and metadata remain relational.
R6 Backfill exports ordered source rows with server-side COPY and leaves the live pointer unchanged.
R7 AGE neighbors use descending weight then ascending ordinal, with per-seed caps and all requested seeds excluded.
R8 SQL remains a separate default backend; AGE failures never select SQL implicitly.
R9 JSONB fields remain only for existing non-topological consumers; the original Open WebUI Function remains unchanged as the SQL reference.
R10 The AGE graph marker must match the relational build before AGE retrieval proceeds.
R11 Snapshot CSV files use internally generated names under PostgreSQL `/tmp/age`, positive IDs (`ord + 1`), and are removed on success or failure.
R12 The image must grant the PostgreSQL OS user write access to `/tmp/age`; AGE loader null score values remain null properties.

CLOSED: Per-edge Cypher endpoint matching is rejected for graph creation: a 10,000-edge batch on 81,473 vertices exceeded three minutes, while AGE's bulk loader loaded and traversed the same shape in 2.0 seconds. AGE does not replace pgvector, relational metadata, or JSONB fields used by active non-traversal consumers. Entity-hop is outside the active /agent path.
Preconditions: PostgreSQL 18, pgvector, AGE 1.7.0, server-side file COPY permission, and source build rows. Failure modes: missing extension, file permission, loader/count mismatch, Cypher errors, and count mismatch abort the graph transaction.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from uuid import uuid4

from psycopg import sql

AGE_GRAPH_PREFIX = "arxiv_section_"
AGE_IMPORT_DIR = "/tmp/age"


def graph_name(build: int) -> str:
    """Return the SQL-safe graph name for one section build.

    Require: `build` is a positive integer database ID.
    Guarantee: a fixed-prefix graph identifier scoped to that build.
    Maintain: caller-supplied text never becomes an SQL or Cypher identifier.
    Assert: the build ID is positive.

    Args:
        build: Section build identity.

    Returns:
        The graph name for `build`.

    Raises:
        AssertionError: If the build ID is not positive.

    Called by:
        Graph creation, catalog checks, and neighbor queries.

    Depends on:
        AGE_GRAPH_PREFIX.
    """
    assert int(build) > 0
    return f"{AGE_GRAPH_PREFIX}{int(build)}"


def _context(conn) -> None:
    """Load AGE and set graph functions in the active transaction.

    Require: AGE shared libraries and extension files are installed.
    Guarantee: AGE functions resolve with ag_catalog first in search_path.
    Maintain: search_path changes are transaction-local.
    Assert: unavailable extension files or privileges fail the caller.

    Args:
        conn: Writable psycopg connection with a transaction in progress.

    Returns:
        None.

    Raises:
        psycopg.Error: If AGE is unavailable or extension setup fails.

    Called by:
        `_age_context`.

    Depends on:
        The installed PostgreSQL AGE extension files.
    """
    conn.execute("CREATE EXTENSION IF NOT EXISTS age")
    conn.execute("LOAD 'age'")
    conn.execute('SET LOCAL search_path = ag_catalog, "$user", public')


@contextmanager
def _age_context(conn) -> Iterator[None]:
    """Load AGE in a savepoint and restore the caller's transaction-local search path.

    Require: the connection can open a transaction and AGE files are installed.
    Guarantee: AGE resolves during the context and the original search_path is restored on success.
    Maintain: failures roll back the savepoint, including AGE setup and search_path changes.
    Assert: caller settings survive AGE operations inside a larger transaction.

    Args:
        conn: Writable psycopg connection.

    Yields:
        None while AGE is loaded with ag_catalog first in search_path.

    Raises:
        psycopg.Error: If AGE setup or search_path restoration fails.

    Called by:
        `_query`, `graph_exists`, and `build_graph`.

    Depends on:
        `_context` and psycopg transaction savepoints.
    """
    with conn.transaction():
        original_search_path = conn.execute("SELECT current_setting('search_path')").fetchone()[0]
        _context(conn)
        yield
        conn.execute("SELECT set_config('search_path', %s, true)", (original_search_path,))


def _query(conn, build: int, statement: str, params: dict, columns: str):
    """Execute an internal Cypher statement with an AGE parameter map.

    Require: statement and output-column declarations are source constants.
    Guarantee: return rows from `ag_catalog.cypher` with values bound as agtype.
    Maintain: graph name and Cypher source are SQL literals; data values remain the third-argument bind parameter.
    Assert: AGE parser or query failures propagate to the caller.

    Args:
        conn: Writable psycopg connection.
        build: Positive section build ID.
        statement: Internal Cypher source, not request text.
        params: JSON-compatible values consumed by Cypher `$name` parameters.
        columns: Internal AGE result-column declaration.

    Returns:
        Rows returned by the Cypher query.

    Raises:
        psycopg.Error: If SQL, AGE parsing, or Cypher execution fails.

    Called by:
        Graph completeness and batch write operations.

    Depends on:
        `_context`, `graph_name`, psycopg.sql, and the age extension.
    """
    delimiter = "$age$"
    suffix = 0
    while delimiter in statement:
        suffix += 1
        delimiter = f"$age{suffix}$"
    cypher_literal = sql.SQL(delimiter) + sql.SQL(statement) + sql.SQL(delimiter)
    query = sql.SQL(
        "SELECT * FROM ag_catalog.cypher({}, {}, %s::ag_catalog.agtype) AS ({})"
    ).format(sql.Literal(graph_name(build)), cypher_literal, sql.SQL(columns))
    with _age_context(conn):
        return conn.execute(
            query,
            (json.dumps(params, separators=(",", ":"), allow_nan=False),),
        ).fetchall()


def graph_exists(conn, build: int) -> bool:
    """Check whether the catalog contains the requested section graph.

    Require: AGE is installed and the caller can read ag_catalog.
    Guarantee: true only for the graph named from `build`.
    Maintain: this catalog read creates no graph state.
    Assert: database errors are not reported as absence.

    Args:
        conn: Writable psycopg connection.
        build: Positive section build ID.

    Returns:
        Whether the AGE catalog contains the graph.

    Raises:
        psycopg.Error: If the AGE catalog cannot be read.

    Called by:
        `require_graph`.

    Depends on:
        `_context` and `graph_name`.
    """
    with _age_context(conn):
        return conn.execute(
            "SELECT 1 FROM ag_catalog.ag_graph WHERE name = %s", (graph_name(build),)
        ).fetchone() is not None


@contextmanager
def _stage_graph_files(conn, build: int) -> Iterator[tuple[str, str]]:
    """Export one relational build to PostgreSQL-local AGE loader files.

    Require: `build` identifies committed or transaction-visible section rows, and the caller can use server COPY.
    Guarantee: yield unique AGE loader basenames for the build's vertices and edges.
    Maintain: PostgreSQL streams source rows directly to disk; no complete edge set enters client memory.
    Assert: every generated path is internal and remains under AGE_IMPORT_DIR.

    Args:
        conn: Writable psycopg connection inside the source transaction.
        build: Positive section build ID.

    Yields:
        Vertex and edge CSV basenames accepted by AGE's file loader.

    Raises:
        psycopg.Error: If source COPY or file cleanup fails.

    Called by:
        `build_graph`.

    Depends on:
        AGE_IMPORT_DIR, psycopg.sql, and PostgreSQL server COPY permissions.
    """
    assert int(build) > 0
    token = uuid4().hex
    vertex_file = f"age_section_{int(build)}_{token}_vertices.csv"
    edge_file = f"age_section_{int(build)}_{token}_edges.csv"
    vertex_path = f"{AGE_IMPORT_DIR}/{vertex_file}"
    edge_path = f"{AGE_IMPORT_DIR}/{edge_file}"
    try:
        with conn.transaction():
            vertex_query = sql.SQL(
                "SELECT ord + 1 AS id, build_id, ord FROM sect_node "
                "WHERE build_id = {} ORDER BY ord"
            ).format(sql.Literal(int(build)))
            conn.execute(sql.SQL(
                "COPY ({}) TO {} WITH (FORMAT csv, HEADER true)"
            ).format(vertex_query, sql.Literal(vertex_path)))

            edge_query = sql.SQL(
                "SELECT a + 1 AS start_id, 'Section' AS start_vertex_type, "
                "b + 1 AS end_id, 'Section' AS end_vertex_type, "
                "weight, dense_sim, sparse_sim, sig FROM sect_edge "
                "WHERE build_id = {} ORDER BY a, b"
            ).format(sql.Literal(int(build)))
            conn.execute(sql.SQL(
                "COPY ({}) TO {} WITH (FORMAT csv, HEADER true)"
            ).format(edge_query, sql.Literal(edge_path)))
        yield vertex_file, edge_file
    finally:
        with conn.transaction():
            for filename in (vertex_file, edge_file):
                program = f"rm -f -- {AGE_IMPORT_DIR}/{filename}"
                conn.execute(sql.SQL("COPY (SELECT 1) TO PROGRAM {}").format(sql.Literal(program)))


def build_graph(conn, build: int, node_count: int, edge_count: int) -> None:
    """Create one complete AGE graph from its transaction-visible relational rows.

    Require: the build's section and edge rows are complete in the caller transaction.
    Guarantee: create one vertex per ordinal, each SIMILAR_TO relationship, and one completion marker.
    Maintain: graph state and source publication commit or roll back together.
    Assert: loaded label row counts equal the declared source counts.

    Args:
        conn: Writable psycopg connection inside the build transaction.
        build: Positive section build ID.
        node_count: Number of contiguous source section ordinals.
        edge_count: Number of source similarity edges.

    Returns:
        None.

    Raises:
        RuntimeError: If source, AGE labels, or loaded graph counts disagree.
        psycopg.Error: If server COPY, the AGE loader, or Cypher fails.

    Called by:
        `section_store._persist` and `rebuild_from_relational`.

    Depends on:
        `_age_context`, `_stage_graph_files`, `_query`, and `graph_name`.
    """
    name = graph_name(build)
    counts = conn.execute(
        "SELECT (SELECT count(*) FROM sect_node WHERE build_id = %s), "
        "(SELECT count(*) FROM sect_edge WHERE build_id = %s)",
        (build, build),
    ).fetchone()
    if tuple(map(int, counts)) != (int(node_count), int(edge_count)):
        raise RuntimeError(
            f"section build {build} declares {(node_count, edge_count)} rows; source has {tuple(counts)}"
        )

    with _stage_graph_files(conn, build) as (vertex_file, edge_file):
        with _age_context(conn):
            if conn.execute("SELECT 1 FROM ag_catalog.ag_graph WHERE name = %s", (name,)).fetchone():
                conn.execute("SELECT ag_catalog.drop_graph(%s, true)", (name,))
            conn.execute("SELECT ag_catalog.create_graph(%s)", (name,))
            conn.execute("SELECT ag_catalog.create_vlabel(%s, 'Section')", (name,))
            conn.execute("SELECT ag_catalog.create_elabel(%s, 'SIMILAR_TO')", (name,))
            conn.execute(
                "SELECT ag_catalog.load_labels_from_file(%s, 'Section', %s, true, true)",
                (name, vertex_file),
            )
            conn.execute(
                "SELECT ag_catalog.load_edges_from_file(%s, 'SIMILAR_TO', %s, true)",
                (name, edge_file),
            )

            vertex_count = conn.execute(sql.SQL("SELECT count(*) FROM {}.{}").format(
                sql.Identifier(name), sql.Identifier("Section"),
            )).fetchone()[0]
            relationship_count = conn.execute(sql.SQL("SELECT count(*) FROM {}.{}").format(
                sql.Identifier(name), sql.Identifier("SIMILAR_TO"),
            )).fetchone()[0]
            if (int(vertex_count), int(relationship_count)) != (int(node_count), int(edge_count)):
                raise RuntimeError(
                    f"AGE graph {name} loaded {(vertex_count, relationship_count)} rows; "
                    f"source declares {(node_count, edge_count)}"
                )

            index_expr = sql.SQL(
                "ag_catalog.agtype_access_operator(VARIADIC ARRAY[properties, '\"ord\"'::ag_catalog.agtype])"
            )
            conn.execute(sql.SQL("CREATE INDEX section_ord_idx ON {}.{} ({})").format(
                sql.Identifier(name), sql.Identifier("Section"), index_expr,
            ))

            marker = _query(
                conn,
                build,
                "CREATE (:GraphBuild {build_id: $build_id, section_count: $section_count, edge_count: $edge_count}) "
                "RETURN count(*)",
                {"build_id": int(build), "section_count": int(node_count), "edge_count": int(edge_count)},
                "written ag_catalog.agtype",
            )
            if not marker or int(str(marker[0][0])) != 1:
                raise RuntimeError(f"AGE graph {name} failed to create its completeness marker")


def rebuild_from_relational(conn, build: int) -> None:
    """Backfill one relational section build without changing its live pointer.

    Require: the section build exists and its section ordinals are contiguous.
    Guarantee: its AGE graph is rebuilt from server-side relational COPY in one transaction.
    Maintain: relational rows and live-build state remain unchanged.
    Assert: source and loaded graph counts match sect_build before commit.

    Args:
        conn: Autocommit psycopg connection.
        build: Existing section build ID.

    Returns:
        None.

    Raises:
        RuntimeError: If source rows or counts are incomplete.
        psycopg.Error: If the transaction, server COPY, or AGE operations fail.

    Called by:
        `section_store.main` when `--build-age-graph` is selected.

    Depends on:
        `build_graph` and `require_graph`.
    """
    with conn.transaction():
        source = conn.execute(
            "SELECT n_nodes, n_edges FROM sect_build WHERE build_id = %s FOR SHARE", (build,)
        ).fetchone()
        if source is None:
            raise RuntimeError(f"section build {build} does not exist")
        node_count, edge_count = map(int, source)
        nodes = conn.execute(
            "SELECT count(*), min(ord), max(ord) FROM sect_node WHERE build_id = %s", (build,)
        ).fetchone()
        if int(nodes[0]) != node_count or (node_count and (nodes[1] != 0 or nodes[2] != node_count - 1)):
            raise RuntimeError(f"section build {build} node rows are not a contiguous declared snapshot")
        build_graph(conn, int(build), node_count, edge_count)
        require_graph(conn, int(build), node_count, edge_count)


def require_graph(conn, build: int, node_count: int, edge_count: int) -> None:
    """Require an AGE graph whose completion marker matches its source build.

    Require: AGE and the relational build are readable.
    Guarantee: return only when one marker has the expected build, node, and edge counts.
    Maintain: validation is read-only and never selects another backend.
    Assert: missing graphs and count mismatches raise RuntimeError.

    Args:
        conn: Writable psycopg connection.
        build: Positive section build ID.
        node_count: Expected number of section vertices.
        edge_count: Expected number of similarity edges.

    Returns:
        None.

    Raises:
        RuntimeError: If the graph is missing or its marker differs.
        psycopg.Error: If AGE catalog access fails.

    Called by:
        `SectionGraph.__init__`, `neighbours`, and backfill.

    Depends on:
        `graph_exists` and `_query`.
    """
    name = graph_name(build)
    if not graph_exists(conn, build):
        raise RuntimeError(f"AGE graph {name} is missing")
    rows = _query(
        conn,
        build,
        "MATCH (m:GraphBuild {build_id: $build_id}) "
        "RETURN m.build_id, m.section_count, m.edge_count",
        {"build_id": int(build)},
        "build_id ag_catalog.agtype, section_count ag_catalog.agtype, edge_count ag_catalog.agtype",
    )
    actual = [tuple(int(str(value)) for value in row) for row in rows]
    expected = [(int(build), int(node_count), int(edge_count))]
    if actual != expected:
        raise RuntimeError(f"AGE graph {name} marker {actual} does not match relational build {expected[0]}")


def neighbours(conn, build: int, seeds: list[int], per_seed: int) -> dict[int, list[dict]]:
    """Return capped, weight-ranked similarity neighbors through Cypher.

    Require: the caller selects AGE and `per_seed` is a nonnegative cap.
    Guarantee: preserve the SQL neighbor fields, stable order, and seed multiplicity.
    Maintain: every requested seed appears in the result, including seeds with no edges.
    Assert: returned seed ordinals belong to the request.

    Args:
        conn: Writable psycopg connection.
        build: Existing section build ID.
        seeds: Ordered section ordinals to expand.
        per_seed: Maximum neighbors for each seed.

    Returns:
        A mapping from seed ordinal to ordered neighbor property rows.

    Raises:
        RuntimeError: If the build or matching AGE graph is absent.
        psycopg.Error: If AGE traversal fails.

    Called by:
        `SectionGraph.neighbours` when its backend is `age`.

    Depends on:
        `require_graph` and `_query`.
    """
    seeds = [int(seed) for seed in seeds]
    out = {seed: [] for seed in seeds}
    if not seeds or per_seed <= 0:
        return out
    row = conn.execute(
        "SELECT n_nodes, n_edges FROM sect_build WHERE build_id = %s", (build,)
    ).fetchone()
    if row is None:
        raise RuntimeError(f"section build {build} does not exist")
    require_graph(conn, build, int(row[0]), int(row[1]))
    query = (
        "UNWIND $seeds AS seed MATCH (s:Section {ord: seed})-[r:SIMILAR_TO]-(n:Section) "
        "WHERE NOT n.ord IN $excluded "
        "WITH seed, n, r ORDER BY seed, r.weight DESC, n.ord "
        "WITH seed, collect({ord: n.ord, weight: r.weight, dense_sim: r.dense_sim, "
        "sparse_sim: r.sparse_sim, sig: r.sig})[0..$limit] AS ranked "
        "UNWIND ranked AS item "
        "RETURN seed, item.ord, item.weight, item.dense_sim, item.sparse_sim, item.sig "
        "ORDER BY seed, item.weight DESC, item.ord"
    )
    rows = _query(
        conn,
        build,
        query,
        {"seeds": seeds, "excluded": seeds, "limit": int(per_seed)},
        "seed ag_catalog.agtype, ord ag_catalog.agtype, weight ag_catalog.agtype, "
        "dense_sim ag_catalog.agtype, sparse_sim ag_catalog.agtype, sig ag_catalog.agtype",
    )
    for seed, ord_, weight, dense_sim, sparse_sim, sig in rows:
        seed_id = int(str(seed))
        assert seed_id in out
        out[seed_id].append({
            "ord": int(str(ord_)),
            "weight": float(str(weight)),
            "dense_sim": None if dense_sim is None else float(str(dense_sim)),
            "sparse_sim": None if sparse_sim is None else float(str(sparse_sim)),
            "sig": int(str(sig)),
        })
    return out
