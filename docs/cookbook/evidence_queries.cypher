// cookbook/evidence_queries.cypher -- the questions a reasoning model asks the
// mirror. One annotated query per digest section; each header states the
// question in plain English and the render_digest row it replaces.
// Spec: .spec/specs/graph-explorer/design.md I12/6.16 (digest, X8-X13 writers)
// Task: playbook.md T15. Cookbook convention: NEVER auto-run; paste into the
// neo4j browser (:7474, db neo4j) or a driver session, substituting $prompt.

// ---------------------------------------------------------------------------
// 1. "What are the strongest highlighted paths for this walk, in words?"
//    Replaces the digest's `pathways: a<->b dwpc|best chain` rows -- with each
//    endpoint's salient terms and source attached so the path reads as text.
MATCH (w:Walk {prompt: $prompt})
MATCH (a:Chunk)-[p:PATHWAY {of: $prompt}]->(b:Chunk)
RETURN a.id, a.source, a.salient, p.dwpc, b.id, b.source, b.salient
ORDER BY p.dwpc DESC
LIMIT 10;

// ---------------------------------------------------------------------------
// 2. "Which correlation chain holds this chunk, and what surrounds it?"
//    Replaces the digest's `chunk_chains: len|ids (correlation-sorted)` row
//    for one chunk of interest -- the dendrite thread it belongs to, in order.
MATCH (c:Chunk {id: $chunk_id})-[e:NEXT_IN_CHAIN {of: $prompt}]-(n)
WITH e.chain AS chain
MATCH (s:Chunk)-[e2:NEXT_IN_CHAIN {of: $prompt, chain: chain}]->(t:Chunk)
RETURN e2.pos, s.id, s.source, s.salient, t.id, t.source, t.salient
ORDER BY e2.pos;

// ---------------------------------------------------------------------------
// 3. "What groups (Louvain communities) did this walk touch, and how hard?"
//    Replaces the digest's `communities: cid|hits/size|sources|keywords` rows.
MATCH (w:Walk {prompt: $prompt})-[t:TOUCHED]->(cs:CommunitySummary)
RETURN cs.cid, t.hits, cs.keywords
ORDER BY t.hits DESC;

// ---------------------------------------------------------------------------
// 4. "Which chunks bridge two sources on a highlighted path?"
//    Replaces reading the pathway rows against the id_key by hand: the
//    cross-register doors (e.g. the Carr-essay -> Bemis hop) fall out directly.
MATCH (a:Chunk)-[p:PATHWAY {of: $prompt}]->(b:Chunk)
WHERE a.source <> b.source
RETURN a.id, a.source, a.salient, p.dwpc, b.id, b.source, b.salient
ORDER BY p.dwpc DESC;

// ---------------------------------------------------------------------------
// 5. "Read a whole chain as terms only -- the story thread in order."
//    Replaces following a `chunk_chains` row term by term with the id_key.
MATCH (s:Chunk)-[e:NEXT_IN_CHAIN {of: $prompt, chain: $chain}]->(t:Chunk)
WITH e.pos AS pos, s, t ORDER BY pos
RETURN collect(DISTINCT s.source + ':' + coalesce(s.salient[0], '?'))
       + [last(collect(t.source + ':' + coalesce(t.salient[0], '?')))]
       AS thread;

// ---------------------------------------------------------------------------
// 6. "What did the walk anchor on, and where did the anchors sit?"
//    Replaces the anchor rows of the walk bundle (Walk-ANCHORS from X10's
//    writer) -- the entry points the evidence grew from.
MATCH (w:Walk {prompt: $prompt})-[:ANCHORS]->(c:Chunk)
RETURN c.id, c.source, c.salient, labels(c) AS communities;
