-- Smoke test for 001_schema.sql. Runs in a transaction and rolls back, so
-- the database is left pristine.  Usage:
--   docker exec -i chunkgraph-pg psql -U graph -d graph -v ON_ERROR_STOP=1 \
--     < tests/test_schema.sql
\set ON_ERROR_STOP on
BEGIN;

\set RUN '11111111-1111-1111-1111-111111111111'
\set RUN2 '22222222-2222-2222-2222-222222222222'

-- ---------------------------------------------------------------- fixture
INSERT INTO graph_run (run_id, label, ingested_at, n_chunks, embed_dim,
                       params, diagnostics)
VALUES (:'RUN', 'smoke', now(), 12, 3,
        '{"k_sigma": 2.0, "has_dense": true}',
        '{"phrases": {"mode": "npmi"}, "vocab": {"mode": "restricted"}}');

INSERT INTO node (run_id, ord, doc_id, chunk_hash, body, attrs)
SELECT :'RUN', g, 'doc' || (g / 4), decode(md5(g::text), 'hex'),
       'chunk body ' || g,
       jsonb_build_object('n_tok', 5, 'qterms', '["alpha","beta"]'::jsonb,
                          'tf', jsonb_build_object('alpha', g + 1, 'beta', 2))
  FROM generate_series(0, 11) g;

-- path 0-1-2-3 plus a cross-doc bridge 1-5
INSERT INTO edge (run_id, src, dst, provenance, strength, sim_sparse, sim_dense,
                  src_doc, dst_doc, valid_from, ingested_at)
VALUES (:'RUN', 0, 1, 'both',   0.90, 0.7, 0.8, 'doc0', 'doc0', now(), now()),
       (:'RUN', 1, 2, 'sparse', 0.80, 0.6, NULL, 'doc0', 'doc0', now(), now()),
       (:'RUN', 2, 3, 'dense',  0.70, 0.1, 0.9, 'doc0', 'doc0', now(), now()),
       (:'RUN', 1, 5, 'both',   0.60, 0.5, 0.5, 'doc0', 'doc1', now(), now());

INSERT INTO community (run_id, cid, members, keywords, medoid, medoid_text)
VALUES (:'RUN', 0, ARRAY[0,1,2,3,5], ARRAY['alpha','beta'], 2, 'chunk body 2');

-- ---------------------------------------------------------------- 1. CHECK
-- edges() only ever emits np.triu, so a reversed pair is a bug: reject it.
DO $$
BEGIN
    INSERT INTO edge (run_id, src, dst, provenance, strength, sim_sparse,
                      src_doc, dst_doc, valid_from, ingested_at)
    VALUES ('11111111-1111-1111-1111-111111111111', 7, 4, 'both', 0.5, 0.5,
            'doc1', 'doc1', now(), now());
    RAISE EXCEPTION 'FAIL: edge_canonical accepted src > dst';
EXCEPTION WHEN check_violation THEN
    RAISE NOTICE 'ok 1: edge_canonical rejects src > dst';
END $$;

-- ---------------------------------------------------------------- 2. FK
DO $$
BEGIN
    INSERT INTO edge (run_id, src, dst, provenance, strength, sim_sparse,
                      src_doc, dst_doc, valid_from, ingested_at)
    VALUES ('11111111-1111-1111-1111-111111111111', 0, 99, 'both', 0.5, 0.5,
            'doc0', 'doc9', now(), now());
    RAISE EXCEPTION 'FAIL: composite FK accepted a dangling ordinal';
EXCEPTION WHEN foreign_key_violation THEN
    RAISE NOTICE 'ok 2: composite FK rejects dangling (run_id, ord)';
END $$;

-- ---------------------------------------------------------------- 3. generated
DO $$
DECLARE s integer;
BEGIN
    SELECT size INTO s FROM community
     WHERE run_id = '11111111-1111-1111-1111-111111111111' AND cid = 0;
    ASSERT s = 5, format('FAIL: generated size = %s, want 5', s);
    RAISE NOTICE 'ok 3: generated size column = %', s;
END $$;

-- ---------------------------------------------------------------- 4. symmetry
DO $$
DECLARE d integer; sym integer;
BEGIN
    SELECT count(*) INTO d FROM edge
     WHERE run_id = '11111111-1111-1111-1111-111111111111';
    SELECT count(*) INTO sym FROM edge_sym
     WHERE run_id = '11111111-1111-1111-1111-111111111111';
    ASSERT sym = 2 * d, format('FAIL: edge_sym %s, want %s', sym, 2 * d);
    RAISE NOTICE 'ok 4: edge_sym doubles % edges -> %', d, sym;
END $$;

-- ---------------------------------------------------------------- 5. traversal
-- 2-hop weighted walk from chunk 0 must reach {0,1,2,5} and NOT 3 (3 hops).
DO $$
DECLARE got integer[];
BEGIN
    WITH RECURSIVE walk AS (
        SELECT '11111111-1111-1111-1111-111111111111'::uuid AS run_id,
               0 AS node, 1.0::double precision AS score, 0 AS hop,
               ARRAY[0] AS path
        UNION ALL
        SELECT w.run_id, s.b, w.score * s.strength, w.hop + 1, w.path || s.b
          FROM walk w
          JOIN edge_sym s ON s.run_id = w.run_id AND s.a = w.node
                         AND s.valid_to IS NULL
         WHERE w.hop < 2 AND NOT s.b = ANY(w.path)
           AND w.score * s.strength > 0.05
    )
    SELECT array_agg(DISTINCT node ORDER BY node) INTO got FROM walk;
    ASSERT got = ARRAY[0,1,2,5], format('FAIL: reached %s, want {0,1,2,5}', got);
    RAISE NOTICE 'ok 5: 2-hop walk reached %', got;
END $$;

-- ---------------------------------------------------------------- 6. jsonb
DO $$
DECLARE n integer;
BEGIN
    SELECT count(*) INTO n FROM node
     WHERE run_id = '11111111-1111-1111-1111-111111111111'
       AND attrs -> 'tf' ? 'alpha';
    ASSERT n = 12, format('FAIL: tf ? alpha matched %s, want 12', n);
    SELECT count(*) INTO n FROM graph_run
     WHERE diagnostics @> '{"phrases": {"mode": "npmi"}}';
    ASSERT n = 1, format('FAIL: diagnostics containment matched %s, want 1', n);
    RAISE NOTICE 'ok 6: jsonb ? and @> operators match';
END $$;

-- ---------------------------------------------------------------- 7. members
DO $$
DECLARE c integer;
BEGIN
    SELECT cid INTO c FROM community
     WHERE run_id = '11111111-1111-1111-1111-111111111111'
       AND members @> ARRAY[3];
    ASSERT c = 0, format('FAIL: members @> ARRAY[3] gave %s', c);
    RAISE NOTICE 'ok 7: community members @> ARRAY[k] works';
END $$;

-- ---------------------------------------------------------------- 8. vector
-- Column ships unconstrained; pg_store stamps the dimension, then HNSW.
DO $$
DECLARE t text; idx integer;
BEGIN
    SELECT format_type(atttypid, atttypmod) INTO t FROM pg_attribute
     WHERE attrelid = 'node_embedding'::regclass AND attname = 'embedding';
    ASSERT t = 'vector', format('FAIL: embedding starts as %s, want vector', t);

    INSERT INTO node_embedding (run_id, ord, embedding)
    VALUES ('11111111-1111-1111-1111-111111111111', 0, '[0.6,0.8,0.0]'),
           ('11111111-1111-1111-1111-111111111111', 1, '[0.0,1.0,0.0]');

    ALTER TABLE node_embedding ALTER COLUMN embedding TYPE vector(3);
    SELECT format_type(atttypid, atttypmod) INTO t FROM pg_attribute
     WHERE attrelid = 'node_embedding'::regclass AND attname = 'embedding';
    ASSERT t = 'vector(3)', format('FAIL: after ALTER got %s', t);

    CREATE INDEX node_embedding_hnsw ON node_embedding
        USING hnsw (embedding vector_ip_ops);
    SELECT count(*) INTO idx FROM pg_indexes
     WHERE indexname = 'node_embedding_hnsw';
    ASSERT idx = 1, 'FAIL: hnsw index not created';
    RAISE NOTICE 'ok 8: vector -> vector(3) + hnsw(vector_ip_ops)';
END $$;

-- inner product ranking on normalized vectors
DO $$
DECLARE best integer;
BEGIN
    SELECT ord INTO best FROM node_embedding
     WHERE run_id = '11111111-1111-1111-1111-111111111111'
     ORDER BY embedding <#> '[0.0,1.0,0.0]'::vector LIMIT 1;
    ASSERT best = 1, format('FAIL: nearest was %s, want 1', best);
    RAISE NOTICE 'ok 9: <#> ranks normalized vectors by cosine';
END $$;

-- ---------------------------------------------------------------- 10. supersede
DO $$
DECLARE closed integer; live integer;
BEGIN
    -- Ingest order matters: retire the incumbent BEFORE inserting the
    -- replacement, or graph_run_live_label rejects the insert.
    closed := supersede_label('smoke', NULL, now());
    ASSERT closed = 4, format('FAIL: closed %s edge intervals, want 4', closed);

    INSERT INTO graph_run (run_id, label, ingested_at, n_chunks, params, diagnostics)
    VALUES ('22222222-2222-2222-2222-222222222222', 'smoke', now(), 12,
            '{}', '{}');

    SELECT count(*) INTO live FROM live_run WHERE label = 'smoke';
    ASSERT live = 1, format('FAIL: %s live runs for label, want 1', live);

    SELECT count(*) INTO live FROM edge
     WHERE run_id = '11111111-1111-1111-1111-111111111111'
       AND valid_to IS NULL;
    ASSERT live = 0, format('FAIL: %s edges still open on old run', live);
    RAISE NOTICE 'ok 10: supersede_label closed % intervals, 1 live run', closed;
END $$;

-- ---------------------------------------------------------------- 11. one live
DO $$
BEGIN
    INSERT INTO graph_run (label, ingested_at, n_chunks, params, diagnostics)
    VALUES ('smoke', now(), 12, '{}', '{}');
    RAISE EXCEPTION 'FAIL: two live runs allowed for one label';
EXCEPTION WHEN unique_violation THEN
    RAISE NOTICE 'ok 11: at most one live run per label';
END $$;

-- ---------------------------------------------------------------- 12. n_chunks
DO $$
BEGIN
    INSERT INTO graph_run (label, ingested_at, n_chunks, params, diagnostics)
    VALUES ('tiny', now(), 9, '{}', '{}');
    RAISE EXCEPTION 'FAIL: n_chunks < 10 accepted';
EXCEPTION WHEN check_violation THEN
    RAISE NOTICE 'ok 12: n_chunks >= 10 enforced (fit asserts the same)';
END $$;

-- ---------------------------------------------------------------- 13. cascade
DO $$
DECLARE n integer;
BEGIN
    DELETE FROM graph_run WHERE run_id = '11111111-1111-1111-1111-111111111111';
    SELECT count(*) INTO n FROM node
     WHERE run_id = '11111111-1111-1111-1111-111111111111';
    ASSERT n = 0, format('FAIL: %s orphan nodes after run delete', n);
    SELECT count(*) INTO n FROM edge
     WHERE run_id = '11111111-1111-1111-1111-111111111111';
    ASSERT n = 0, format('FAIL: %s orphan edges after run delete', n);
    RAISE NOTICE 'ok 13: deleting a run cascades to nodes/edges/communities';
END $$;

ROLLBACK;
