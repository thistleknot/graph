-- ChunkGraph -> Postgres. Hybrid model: typed columns for traversal keys,
-- jsonb for open-ended payload. Target: pgvector/pgvector:0.8.1-pg18-trixie
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- edges() emits exactly these three provenance values via PROV
CREATE TYPE edge_provenance AS ENUM ('sparse', 'dense', 'both');

-- ---------------------------------------------------------------- run
-- One fit() == one run. Chunk ordinals are only meaningful within a run,
-- so every table below is run-scoped.
CREATE TABLE graph_run (
    run_id        uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    label         text        NOT NULL,
    ingested_at   timestamptz NOT NULL,          -- cg.ingested_at
    n_chunks      integer     NOT NULL CHECK (n_chunks >= 10),  -- fit() asserts >=10
    embed_dim     integer,                       -- NULL when embed_fn is None
    params        jsonb       NOT NULL DEFAULT '{}'::jsonb,   -- k_sigma, hops, lam_div...
    diagnostics   jsonb       NOT NULL DEFAULT '{}'::jsonb,   -- cg.diagnostics verbatim
    superseded_at timestamptz
);
-- At most one live run per label; re-ingest supersedes rather than deletes.
CREATE UNIQUE INDEX graph_run_live_label
    ON graph_run (label) WHERE superseded_at IS NULL;
CREATE INDEX graph_run_diag_gin
    ON graph_run USING gin (diagnostics jsonb_path_ops);

-- ---------------------------------------------------------------- node
CREATE TABLE node (
    run_id     uuid    NOT NULL REFERENCES graph_run ON DELETE CASCADE,
    ord        integer NOT NULL,                 -- chunk index i
    doc_id     text    NOT NULL,                 -- cg.doc_id[i]  (R8)
    chunk_hash bytea   NOT NULL,                 -- stable cross-run identity
    body       text    NOT NULL,                 -- cg.chunks[i]
    -- {"n_tok":int,"qterms":[...],"disc":[...],"tf":{term:freq}}
    attrs      jsonb   NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (run_id, ord)
);
CREATE INDEX node_doc      ON node (run_id, doc_id);
CREATE INDEX node_hash     ON node (chunk_hash);          -- same chunk across runs
CREATE INDEX node_attrs_gin ON node USING gin (attrs jsonb_path_ops);
-- term -> chunks, straight off the jsonb tf map
CREATE INDEX node_tf_gin   ON node USING gin ((attrs -> 'tf'));
CREATE INDEX node_body_trgm ON node USING gin (body gin_trgm_ops);

-- ---------------------------------------------------------------- edge
-- edges() walks np.triu(A,1), so src < dst always holds: one row per
-- undirected edge. The CHECK makes that invariant the database's problem.
CREATE TABLE edge (
    run_id      uuid    NOT NULL,
    src         integer NOT NULL,
    dst         integer NOT NULL,
    provenance  edge_provenance  NOT NULL,
    strength    double precision NOT NULL,
    sim_sparse  double precision NOT NULL,
    sim_dense   double precision,                -- NULL when no embed_fn
    src_doc     text    NOT NULL,
    dst_doc     text    NOT NULL,
    attrs       jsonb   NOT NULL DEFAULT '{}'::jsonb,
    valid_from  timestamptz NOT NULL,
    valid_to    timestamptz,                     -- NULL until superseded (R11)
    ingested_at timestamptz NOT NULL,
    PRIMARY KEY (run_id, src, dst),
    CONSTRAINT edge_canonical CHECK (src < dst),
    CONSTRAINT edge_interval  CHECK (valid_to IS NULL OR valid_to >= valid_from),
    FOREIGN KEY (run_id, src) REFERENCES node (run_id, ord) ON DELETE CASCADE,
    FOREIGN KEY (run_id, dst) REFERENCES node (run_id, ord) ON DELETE CASCADE
);
-- Traversal indexes. Partial on live rows: the recursive CTE never pays for
-- superseded history. INCLUDE gives index-only hops.
CREATE INDEX edge_live_src ON edge (run_id, src, strength DESC)
    INCLUDE (dst) WHERE valid_to IS NULL;
CREATE INDEX edge_live_dst ON edge (run_id, dst, strength DESC)
    INCLUDE (src) WHERE valid_to IS NULL;
CREATE INDEX edge_prov     ON edge (run_id, provenance) WHERE valid_to IS NULL;
CREATE INDEX edge_bridge   ON edge (run_id, src_doc, dst_doc) WHERE valid_to IS NULL;
CREATE INDEX edge_attrs_gin ON edge USING gin (attrs jsonb_path_ops);
-- as-of queries
CREATE INDEX edge_temporal ON edge (run_id, valid_from, valid_to);

-- Undirected adjacency. Traversal reads this, never `edge` directly.
CREATE VIEW edge_sym AS
SELECT run_id, src AS a, dst AS b, strength, provenance,
       sim_sparse, sim_dense, src_doc AS a_doc, dst_doc AS b_doc,
       valid_from, valid_to
  FROM edge
UNION ALL
SELECT run_id, dst, src, strength, provenance,
       sim_sparse, sim_dense, dst_doc, src_doc,
       valid_from, valid_to
  FROM edge;

-- ---------------------------------------------------------------- community
CREATE TABLE community (
    run_id      uuid    NOT NULL REFERENCES graph_run ON DELETE CASCADE,
    cid         integer NOT NULL,                -- Louvain partition id, run-local
    members     integer[] NOT NULL,
    size        integer GENERATED ALWAYS AS (cardinality(members)) STORED,
    keywords    text[]  NOT NULL,                -- top-5 tf*idf
    medoid      integer NOT NULL,
    medoid_text text,
    attrs       jsonb   NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (run_id, cid),
    FOREIGN KEY (run_id, medoid) REFERENCES node (run_id, ord) ON DELETE CASCADE
);
-- "which community holds chunk k" -> members @> ARRAY[k]
CREATE INDEX community_members_gin ON community USING gin (members);
CREATE INDEX community_kw_gin      ON community USING gin (keywords);
CREATE INDEX community_size        ON community (run_id, size DESC);

-- ---------------------------------------------------------------- embedding
-- cg.E is already L2-normalized, so inner product == cosine; vector_ip_ops
-- is the cheaper operator class. Dimension is stamped by pg_store.py at
-- first ingest, because it depends on model_dir.
CREATE TABLE node_embedding (
    run_id    uuid    NOT NULL,
    ord       integer NOT NULL,
    embedding vector  NOT NULL,
    PRIMARY KEY (run_id, ord),
    FOREIGN KEY (run_id, ord) REFERENCES node (run_id, ord) ON DELETE CASCADE
);

-- ---------------------------------------------------------------- supersede
-- Re-ingest under an existing label: close the old run's edge intervals and
-- retire the run. History stays queryable via the as-of pattern.
--
-- p_keep IS NULL retires every live run under the label. Callers MUST use
-- that form BEFORE inserting the replacement row, because
-- graph_run_live_label permits only one live run per label -- inserting
-- first and superseding afterwards deadlocks on that index.
CREATE FUNCTION supersede_label(p_label text, p_keep uuid, p_at timestamptz)
RETURNS integer LANGUAGE plpgsql AS $$
DECLARE n integer;
BEGIN
    UPDATE edge e SET valid_to = p_at
      FROM graph_run r
     WHERE e.run_id = r.run_id
       AND r.label  = p_label
       AND (p_keep IS NULL OR r.run_id <> p_keep)
       AND e.valid_to IS NULL;
    GET DIAGNOSTICS n = ROW_COUNT;

    UPDATE graph_run SET superseded_at = p_at
     WHERE label = p_label
       AND (p_keep IS NULL OR run_id <> p_keep)
       AND superseded_at IS NULL;
    RETURN n;
END $$;

-- Convenience: the live run for a label.
CREATE VIEW live_run AS
SELECT * FROM graph_run WHERE superseded_at IS NULL;
