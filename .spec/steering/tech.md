---
inclusion: always
---

# Tech

## Language & runtime

- Python 3.10, Windows primary workstation.
- Postgres 18 in Docker: `pgvector/pgvector:0.8.1-pg18-trixie`.

## Dependencies

**Required:** numpy, scipy (stats, sparse, interpolate), psycopg 3 (`psycopg.types.json.Jsonb`), networkx, pgvector.

**Optional, must degrade gracefully:** nltk.stem (stemming), gensim (R9 phrases), wordfreq (`background='wordfreq'` only), matplotlib (DRAW), sentence-transformers / local MiniLM (DENSE; absent ⇒ sparse-only per R5).

## Connection

`CHUNKGRAPH_DSN`, default `postgresql://graph:graph@localhost:5433/graph`.
Host port is **5433** — 5432 on this machine is held by the `langchain_postgres`
container running the same image.

## Hard constraints

These are paid-for. Violating them has already cost time.

1. **Volume mount is `/var/lib/postgresql`, NOT `/var/lib/postgresql/data`.**
   PG18+ images store data in a major-version subdirectory. Mounting at `data`
   yields an ICU/locale collation failure at init.
2. **`./sql` is migrations-only.** Every `*.sql` there runs once, alphabetically,
   on an empty data dir. Query snippets with psql placeholders live in
   `./cookbook` so they never execute at init.
3. **jsonb is payload-only.** `attrs` never holds a traversal or join key.
   `src`/`dst` are real indexed integer columns — that is what makes k-hop
   expansion an index scan instead of full-table GIN work. A "tidy-up" that
   moves a traversal key into jsonb is a regression.
4. **Embedding dimension is immutable after first ingest.** `pg_store.py` stamps
   it from `model_dir`; `ALTER COLUMN embedding vector(d)` against existing rows
   is a hazard. Changing models means a new run, not a migration. Pinned by test.
5. **No materialized n×n dense product** (R10) — thresholding happens inside the
   block loop. A dense product exhausts memory at ~18k chunks.
6. **Louvain membership is never LLM-assigned.** Partition is deterministic;
   models may only *name* an already-frozen cluster.

## Postgres tuning (docker-compose)

`shm_size: 1gb` (default 64MB is tight for Louvain-sized sorts and HNSW builds),
`maintenance_work_mem=1GB` (HNSW), `work_mem=64MB` (recursive CTE hash joins),
`jit=off` (net loss on short analytic queries).

`vector_ip_ops` is the operator class — `cg.E` rows are already L2-normalized, so
inner product equals cosine and is cheaper.

## Banned

- **No LLM in the graph construction path.** Chunking, edges, fusion, and
  community membership are deterministic. *(Inferred from design, now explicit.)*
