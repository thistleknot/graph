Using pure PostgreSQL as a property graph database without any extensions is a highly viable strategy. By utilizing `jsonb` columns for properties, you retain the schema-less flexibility of traditional graph databases like Neo4j, while keeping the transactional reliability (ACID compliance) of Postgres. [1, 2, 3, 4, 5]

---

## 1. The Schema Design

The most efficient setup uses two tables: one for nodes (vertices) and one for edges (relationships). [2, 6]

```sql
-- Create a table for Nodes
CREATE TABLE nodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    label TEXT NOT NULL,                     -- e.g., 'Person', 'Company'
    properties JSONB DEFAULT '{}'::jsonb NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create a table for Edges
CREATE TABLE edges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id UUID NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    target_id UUID NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    label TEXT NOT NULL,                     -- e.g., 'KNOWS', 'WORKS_AT'
    properties JSONB DEFAULT '{}'::jsonb NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Optional: Prevent duplicate directed edges of the same type between two nodes
    CONSTRAINT unique_edge UNIQUE (source_id, target_id, label)
);
```

---

## 2. Indexes (Crucial for Performance)

Without native graph indexing, you must optimize lookup and traversal speeds via relational indexes. [7, 8]

```sql
-- Standard B-Tree indexes for fast traversal lookups
CREATE INDEX idx_nodes_label ON nodes(label);
CREATE INDEX idx_edges_source ON edges(source_id);
CREATE INDEX idx_edges_target ON edges(target_id);
CREATE INDEX idx_edges_label ON edges(label);

-- GIN (Generalized Inverted Index) for searching inside the JSONB fields
CREATE INDEX idx_nodes_props ON nodes USING gin (properties);
CREATE INDEX idx_edges_props ON edges USING gin (properties);
```

---

## 3. Querying the Graph (Graph Traversal)

To walk the graph (e.g., finding connections multiple hops away), use a `WITH RECURSIVE` Common Table Expression (CTE). [1, 2, 6]

### Example: Find "Friends of Friends" (2 Hops Away) up to Max Depth

If you want to start from a specific `node_id` and find everyone connected via the `KNOWS` relationship:

```sql
WITH RECURSIVE graph_walk AS (
    -- Anchor member: Initial edges coming from the starting node
    SELECT
        e.target_id AS current_node_id,
        1 AS depth,
        ARRAY[e.source_id, e.target_id] AS path
    FROM edges e
    WHERE e.source_id = 'STARTING_NODE_UUID_HERE'
      AND e.label = 'KNOWS'

    UNION ALL

    -- Recursive member: Find the next hop
    SELECT
        e.target_id,
        gw.depth + 1,
        gw.path || e.target_id
    FROM graph_walk gw
    JOIN edges e ON gw.current_node_id = e.source_id
    WHERE e.label = 'KNOWS'
      AND gw.depth < 3  -- Limit depth to avoid infinite loops / slow queries
      AND NOT (e.target_id = ANY(gw.path)) -- Cycle prevention: don't revisit nodes
)
SELECT DISTINCT
    n.id,
    n.label,
    n.properties,
    gw.depth
FROM graph_walk gw
JOIN nodes n ON gw.current_node_id = n.id;
```

---

## 4. Querying and Mutating `jsonb` Properties

### Inserting Data

```sql
-- Insert a node with unstructured data
INSERT INTO nodes (label, properties)
VALUES (
    'Person',
    '{"name": "Alice", "age": 30, "skills": ["Postgres", "SQL"]}'
);

-- Insert an edge with properties
INSERT INTO edges (source_id, target_id, label, properties)
VALUES (
    'NODE_A_UUID',
    'NODE_B_UUID',
    'KNOWS',
    '{"since": 2021, "closeness": "high"}'
);
```

### Filtering Nodes by Properties

Using the `@>` containment operator (which takes advantage of the GIN index):

```sql
-- Find all people who know Postgres
SELECT *
FROM nodes
WHERE label = 'Person'
  AND properties @> '{"skills": ["Postgres"]}';
```

---

## Pros and Cons of This Approach

| Feature | Postgres + JSONB (Pure Relational) | Dedicated Graph / Apache AGE |
|---|---|---|
| Query Syntax | Verbose (SQL CTEs / Joins) | Concise (openCypher MATCH clauses) |
| Ecosystem Maturity | Exceptionally high; works out-of-the-box everywhere. | Requires specific modules/extension management. |
| Depth Scaling | Slower at 4+ depth layers due to Join explosions. | Optimized for deep pointer-jumping traversals. |
| ACID Integrity | 100% Native. | Dependent on extension integration. |

Would you like help writing specific SQL queries for your use case (like finding the shortest path or tracking acyclic directed paths), or would you like to see how to structure foreign keys for concrete type enforcement? [8, 9] 

[1] [https://www.reddit.com](https://www.reddit.com/r/programming/comments/124xsci/postgres_the_graph_database_you_didnt_know_you_had/)
[2] [https://micelclaw.com](https://micelclaw.com/blog/knowledge-graph-postgresql/)
[3] [https://mohammadshaker.com](https://mohammadshaker.com/en/blog/neo4j-vs-postgres-age-vs-jsonl-graph-data-showdown)
[4] [https://www.puppygraph.com](https://www.puppygraph.com/blog/postgresql-graph-database)
[5] [https://www.youtube.com](https://www.youtube.com/watch?v=LqLiA4fpLvc&t=104)
[6] [https://kushankurdas.medium.com](https://kushankurdas.medium.com/using-postgresql-as-a-graph-database-a-simple-approach-for-beginners-c76d3bc9e82c)
[7] [https://www.klioba.com](https://www.klioba.com/postgresql-as-a-graph-database)
[8] [https://news.ycombinator.com](https://news.ycombinator.com/item?id=10316872)
[9] [https://www.youtube.com](https://www.youtube.com/watch?v=8q3Vl_hCtCI&t=584)