louvain_pg.py -- Louvain communities over the pure-Postgres property graph (nodes/edges, jsonb).

STAGED PIPELINE
| # | Stage   | Step                                                          | Tag         | Guards |
|---|---------|---------------------------------------------------------------|-------------|--------|
| 1 | INGEST  | SQL: collapse edges to undirected pairs, sum weight           | [mandatory] | R1,R2  |
| 2 | BUILD   | networkx.Graph from pairs (UUID -> str node keys)             | [mandatory] | R3     |
| 3 | DETECT  | networkx.community.louvain_communities(seed, resolution)      | [mandatory] | R4,R5  |
| 4 | CHECK   | assert partition covers every node exactly once               | [contract]  | R6     |
| 5 | PERSIST | upsert node_community(run_id, node_id, community_id)          | [mandatory] | R7     |
| 6 | SPLIT   | --connected: split communities into connected components      | [opt]       | R8     |

GUARDS (EARS)
R1  WHEN edges are read, THE SYSTEM SHALL treat (a,b) and (b,a) as one undirected pair.
R2  WHEN an edge has no properties.weight, THE SYSTEM SHALL use weight 1.0.
R3  IF an edge references a node absent from the node set, THEN THE SYSTEM SHALL fail fast.
R4  THE SYSTEM SHALL pass a fixed seed so reruns give the same partition.
R5  WHERE --resolution is set, THE SYSTEM SHALL forward it (higher = smaller communities).
R6  IF the partition is not a disjoint cover of all nodes, THEN THE SYSTEM SHALL raise.
R7  THE SYSTEM SHALL write all assignments in one transaction keyed by run_id.
R8  WHERE --connected is set, THE SYSTEM SHALL split any disconnected community.

CLOSED
- Louvain inside a WITH RECURSIVE CTE: rejected. Modularity optimisation is an iterative
  local-move + aggregate loop; recursive CTEs only do reachability. Run it client-side.
- Louvain on the directed graph: rejected. Standard modularity here is the undirected form.

PRECONDITIONS: tables nodes(id uuid, label, properties jsonb), edges(source_id, target_id,
label, properties jsonb) exist; pip install psycopg networkx.
FAILURE MODES: Louvain can emit internally disconnected communities (Traag et al. 2019,
Leiden paper); use --connected, or swap in leidenalg/igraph for a guaranteed-connected result.
"""
import argparse
import uuid

import networkx as nx
import psycopg

DDL = """
CREATE TABLE IF NOT EXISTS node_community (
    run_id       UUID NOT NULL,
    node_id      UUID NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    community_id INT  NOT NULL,
    PRIMARY KEY (run_id, node_id)
);
CREATE INDEX IF NOT EXISTS idx_node_community_comm ON node_community(run_id, community_id);
"""

EDGE_SQL = """
SELECT LEAST(source_id, target_id)::text    AS a,
       GREATEST(source_id, target_id)::text AS b,
       SUM(COALESCE((properties->>'weight')::float, 1.0)) AS w
FROM edges
WHERE (%(label)s::text IS NULL OR label = %(label)s)
  AND source_id <> target_id
GROUP BY 1, 2
"""


def fetch_edges(conn, label):
    """Require: open connection. Guarantee: list of (a, b, w), one row per undirected pair."""
    with conn.cursor() as cur:
        cur.execute(EDGE_SQL, {"label": label})
        return cur.fetchall()


def fetch_node_ids(conn):
    """Guarantee: set of all node ids as str."""
    with conn.cursor() as cur:
        cur.execute("SELECT id::text FROM nodes")
        return {r[0] for r in cur.fetchall()}


def build_graph(node_ids, edges):
    """Require: node_ids set, edges list. Guarantee: Graph with isolates included. Assert R3."""
    g = nx.Graph()
    g.add_nodes_from(node_ids)
    for a, b, w in edges:
        assert a in node_ids and b in node_ids, f"edge references unknown node: {a}, {b}"
        g.add_edge(a, b, weight=w)
    return g


def split_disconnected(g, communities):
    """Guarantee: every returned community induces a connected subgraph (R8)."""
    out = []
    for c in communities:
        out.extend(set(cc) for cc in nx.connected_components(g.subgraph(c)))
    return out


def detect(g, resolution, seed, connected):
    """Require: g non-empty. Guarantee: list of disjoint node sets covering g (R4, R5, R6, R8)."""
    comms = nx.community.louvain_communities(g, weight="weight", resolution=resolution, seed=seed)
    if connected:
        comms = split_disconnected(g, comms)
    seen = set()
    for c in comms:
        assert not (seen & c), "communities overlap"
        seen |= c
    assert seen == set(g.nodes), "partition does not cover all nodes"
    return comms


def persist(conn, run_id, communities):
    """Guarantee: all rows written in one transaction (R7)."""
    rows = [(run_id, node, i) for i, c in enumerate(sorted(communities, key=len, reverse=True))
            for node in c]
    with conn.cursor() as cur:
        cur.execute(DDL)
        cur.executemany(
            "INSERT INTO node_community (run_id, node_id, community_id) "
            "VALUES (%s, %s::uuid, %s) "
            "ON CONFLICT (run_id, node_id) DO UPDATE SET community_id = EXCLUDED.community_id",
            rows,
        )
    conn.commit()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dsn", required=True)
    p.add_argument("--label", default=None, help="edge label filter; omit for all edges")
    p.add_argument("--resolution", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--connected", action="store_true")
    a = p.parse_args()

    with psycopg.connect(a.dsn) as conn:
        g = build_graph(fetch_node_ids(conn), fetch_edges(conn, a.label))
        comms = detect(g, a.resolution, a.seed, a.connected)
        run_id = uuid.uuid4()
        persist(conn, run_id, comms)
    q = nx.community.modularity(g, comms, weight="weight")
    print(f"run_id={run_id} nodes={g.number_of_nodes()} edges={g.number_of_edges()} "
          f"communities={len(comms)} modularity={q:.4f}")


if __name__ == "__main__":
    main()
