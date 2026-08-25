-- Query patterns for the ChunkGraph store. Not run at init; a cookbook.
-- :label is the graph_run label.

-- 1. Weighted k-hop expansion from a seed chunk. Mirrors _gist_walk: hop
--    through edge_sym, decay by strength, keep the best path to each node.
--    Cycle-safe via the visited array.
WITH RECURSIVE seed AS (
    SELECT r.run_id, n.ord
      FROM live_run r JOIN node n USING (run_id)
     WHERE r.label = :'label' AND n.ord = :seed
), walk AS (
    SELECT run_id, ord AS node, 1.0::double precision AS score,
           0 AS hop, ARRAY[ord] AS path
      FROM seed
    UNION ALL
    SELECT w.run_id, s.b, w.score * s.strength, w.hop + 1, w.path || s.b
      FROM walk w
      JOIN edge_sym s ON s.run_id = w.run_id AND s.a = w.node
                     AND s.valid_to IS NULL
     WHERE w.hop < 2                      -- cg.H
       AND NOT s.b = ANY(w.path)
       AND w.score * s.strength > 0.05
)
SELECT node, max(score) AS score, min(hop) AS hop
  FROM walk GROUP BY node ORDER BY score DESC LIMIT 25;

-- 2. Strongest neighbours of a chunk, with text. Index-only on edge_live_*.
SELECT s.b, s.strength, s.provenance, n.doc_id, left(n.body, 120) AS preview
  FROM live_run r
  JOIN edge_sym s USING (run_id)
  JOIN node n ON n.run_id = s.run_id AND n.ord = s.b
 WHERE r.label = :'label' AND s.a = :seed AND s.valid_to IS NULL
 ORDER BY s.strength DESC LIMIT 10;

-- 3. Which community holds a chunk (GIN on members).
SELECT c.cid, c.size, c.keywords, c.medoid_text
  FROM live_run r JOIN community c USING (run_id)
 WHERE r.label = :'label' AND c.members @> ARRAY[:seed];

-- 4. Hybrid retrieval: pgvector ANN for the anchors, graph expansion for the
--    neighbourhood. embedding is L2-normalized, so <#> (negative inner
--    product) ranks identically to cosine and is cheaper.
WITH anchors AS (
    SELECT ne.ord, -(ne.embedding <#> :'qvec'::vector) AS sim
      FROM live_run r JOIN node_embedding ne USING (run_id)
     WHERE r.label = :'label'
     ORDER BY ne.embedding <#> :'qvec'::vector
     LIMIT 2                              -- cg.query k_anchor
), expanded AS (
    SELECT a.ord AS node, a.sim AS score FROM anchors a
    UNION ALL
    SELECT s.b, a.sim * s.strength
      FROM anchors a
      JOIN live_run r ON r.label = :'label'
      JOIN edge_sym s ON s.run_id = r.run_id AND s.a = a.ord
                     AND s.valid_to IS NULL
)
SELECT e.node, max(e.score) AS score, left(n.body, 160) AS preview
  FROM expanded e
  JOIN live_run r ON r.label = :'label'
  JOIN node n ON n.run_id = r.run_id AND n.ord = e.node
 GROUP BY e.node, n.body ORDER BY score DESC LIMIT 20;

-- 5. jsonb: chunks containing a term, read off the tf map (node_tf_gin).
SELECT n.ord, n.doc_id, (n.attrs -> 'tf' ->> :'term')::int AS tf
  FROM live_run r JOIN node n USING (run_id)
 WHERE r.label = :'label' AND n.attrs -> 'tf' ? :'term'
 ORDER BY tf DESC LIMIT 20;

-- 6. Cross-document bridges: edges whose endpoints come from different
--    source docs. These are the interesting ones for synthesis.
SELECT e.src, e.dst, e.src_doc, e.dst_doc, e.strength, e.provenance
  FROM live_run r JOIN edge e USING (run_id)
 WHERE r.label = :'label' AND e.valid_to IS NULL AND e.src_doc <> e.dst_doc
 ORDER BY e.strength DESC LIMIT 25;

-- 7. Bitemporal as-of: the graph as it stood at an instant, superseded runs
--    included.
SELECT e.src, e.dst, e.strength
  FROM graph_run r JOIN edge e USING (run_id)
 WHERE r.label = :'label'
   AND e.valid_from <= :'asof'::timestamptz
   AND (e.valid_to IS NULL OR e.valid_to > :'asof'::timestamptz);

-- 8. Where dense and sparse disagree — edges one channel found and the other
--    did not. Useful for auditing the blend.
SELECT provenance, count(*), avg(strength)::numeric(6,4) AS avg_strength,
       avg(sim_sparse)::numeric(6,4) AS avg_sparse,
       avg(sim_dense)::numeric(6,4)  AS avg_dense
  FROM live_run r JOIN edge e USING (run_id)
 WHERE r.label = :'label' AND e.valid_to IS NULL
 GROUP BY provenance ORDER BY count(*) DESC;

-- 9. Run diagnostics by jsonb containment (graph_run_diag_gin).
SELECT label, ingested_at, n_chunks, diagnostics
  FROM graph_run
 WHERE diagnostics @> '{"phrases": {"mode": "npmi"}}'
 ORDER BY ingested_at DESC;
