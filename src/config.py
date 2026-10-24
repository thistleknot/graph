"""Environment configuration shared by graph and BM25 services.

| Stage | Mechanism | Tag | Ref |
|---|---|---|---|
| CONFIGURE | Resolve separate graph and BM25 database DSNs. | [mandatory] | R1 |

R1 `CHUNKGRAPH_DSN` owns graph storage; `ARXIV_BM25_DSN` owns BM25 storage and has an independent deployment default.

Preconditions: database endpoints are reachable by the caller. Failure modes: unavailable DSNs fail at connect time.
"""
import os

# ---- ports (the only place a port number is written)
PG_PORT         = int(os.environ.get("CHUNKGRAPH_PG_PORT", "5433"))
NEO4J_HTTP_PORT = int(os.environ.get("NEO4J_HTTP_PORT", "7474"))
NEO4J_BOLT_PORT = int(os.environ.get("NEO4J_BOLT_PORT", "7687"))

# ---- postgres
DSN = os.environ.get("CHUNKGRAPH_DSN",
                     f"postgresql://graph:graph@localhost:{PG_PORT}/graph")
ARXIV_BM25_DSN = os.environ.get(
    "ARXIV_BM25_DSN",
    "postgresql://langchain:langchain@localhost:5432/langchain",
)
ARXIV_BM25_SAMPLE_SIZE = int(os.environ.get("ARXIV_BM25_SAMPLE_SIZE", "80000"))
ARXIV_BM25_SAMPLE_SEED = os.environ.get("ARXIV_BM25_SAMPLE_SEED", "arxiv-bm25-sample-2026-10")

# ---- neo4j (local container runs NEO4J_AUTH=none)
NEO4J_HTTP    = os.environ.get("NEO4J_HTTP", f"http://localhost:{NEO4J_HTTP_PORT}")
NEO4J_BOLT    = os.environ.get("NEO4J_BOLT", f"bolt://localhost:{NEO4J_BOLT_PORT}")
NEO4J_DB      = os.environ.get("NEO4J_DB", "neo4j")
NEO4J_BROWSER = f"{NEO4J_HTTP}/browser/?preselectAuthMethod=NO_AUTH"
# DEPRECATED per 6.21(b): auth is None unless BOTH env vars are set. The
# hardcoded ("neo4j", "graphgraph") pair is gone from the tree.
NEO4J_AUTH = ((os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"])
              if os.environ.get("NEO4J_USER") and os.environ.get("NEO4J_PASSWORD")
              else None)

# ---- models: ONE default (6.21(b) winner), expanded per-user
MODEL_DIR = os.environ.get("CHUNKGRAPH_MODEL_DIR") or \
            os.path.expanduser("~/models/m2v-minilm-l6-256")

# ---- caches
CACHE_DIR = os.environ.get("CHUNKGRAPH_CACHE",
                           os.path.join(os.path.expanduser("~"), ".cache", "chunkgraph"))
