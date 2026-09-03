"""
export_neo4j.py -- emit a persisted ChunkGraph run as neo4j-admin bulk-import CSV.

WHY THIS SHAPE
Records are formed around chunks of text, not around nodes and edges. The term
side of the graph is therefore an INVERTED INDEX (term -> record_ids), not a
stored edge list, and chunk->term relationships are DERIVED here rather than
materialized in Postgres. Neo4j's bulk importer wants explicit relationship
rows, so the derivation happens at export time -- once, into CSV -- instead of
being carried as a second copy of the same facts in the database.

    CONTAINS  (:Chunk)-[:CONTAINS]->(:Term)    derived from attrs->'tf'
    SIMILAR   (:Chunk)-[:SIMILAR]->(:Chunk)    read from the edge table

`(:Chunk)-[:CONTAINS]->(:Term)<-[:CONTAINS]-(:Chunk)` is the term-mediated hop,
native in Cypher, with no chunk-chunk edge stored for it.

WHY SIMILAR IS STILL EXPORTED
Sparse SIMILAR edges are redundant with the CONTAINS pattern above -- they are a
summary of it. Dense ones are NOT: embedding proximity is not a function of
shared terms, so dropping them would silently make the export sparse-only.
--edges controls this; the default keeps both and the redundancy is deliberate,
because a consumer that only walks SIMILAR should still see the whole graph.

Require:   a live run under `label`; read-only DB access.
Guarantee: chunks.csv + terms.csv + contains.csv + similar.csv + import.sh in
           `out`, neo4j-admin header format, every :START_ID/:END_ID resolvable
           within its id-space. chunks.csv carries a `source` column, empty on
           runs predating R20 (design.md §6.14 R20). vector_index.cypher is
           additionally emitted when at least one embedding was written.
Maintain:  no writes; no re-partitioning; cid is read, never recomputed.
           `write_walk` additionally guarantees one (:Walk) per prompt with
           its ANCHORS and PATHWAY edges written, or an exception.
           `write_digest` additionally guarantees NEXT_IN_CHAIN, Chunk.salient
           and CommunitySummary/TOUCHED are written for an existing (:Walk),
           or an exception (X11-X15).

GUARDS (EARS)
X1 Chunk ids and Term ids SHALL occupy SEPARATE neo4j id-spaces. A term whose
   text is "4471" and a record_id 4471 are different nodes; sharing one space
   silently MERGES them and the importer reports success.
X2 WHERE a term appears in many chunks, it SHALL be emitted as ONE Term node.
   Terms are corpus-scoped; chunks are not.
X3 Every :START_ID and :END_ID SHALL resolve to an emitted node id. Counted and
   asserted before the files are written, because neo4j-admin fails the whole
   import on the first dangling reference.
X4 Text SHALL be emitted through csv.writer, never hand-quoted. Chunk bodies
   contain commas, quotes and newlines.
X5 The run label and run_id SHALL be recorded in import.sh, so an imported
   database can name the run it came from.
X6 Nodes SHALL be emitted one id-space per file. A single file carrying two
   `:ID(...)` columns is rejected by current neo4j-admin; the 4-file layout is
   what makes X1 structural rather than a per-row invariant.
X7 The embedding column SHALL be `;`-separated tokens copied verbatim from
   storage, of uniform length across the run, and `vector_index.cypher` SHALL
   be emitted if and only if at least one vector was written.
X8 A chunk with a community SHALL carry `C<cid>` as a second label in the
   `:LABEL` column, so the imported database is partitionable without a
   post-import pass.
X9  The live writer SHALL MATCH Chunk nodes, never MERGE them, and SHALL address
    them by `id` as a STRING -- the CSV import runs --id-type=STRING, so an integer
    parameter matches nothing and the write silently succeeds having written nothing.
X10 A walk SHALL be re-writable: (:Walk) is keyed on `prompt`, (:PATHWAY) on
    (src, dst, of) with the ordinal pair normalized low->high, so a second write of
    the same walk updates in place instead of duplicating. Any transport or MATCH
    shortfall SHALL raise; there is no partial-write fallback.
X11 A digest re-write SHALL NOT leave stale NEXT_IN_CHAIN edges: unlike PATHWAY
    (X10), the chain set is a function of the current dendrite cut and can
    shrink or reorder between writes, so the writer SHALL delete all
    NEXT_IN_CHAIN edges for `of=prompt` before re-merging the current set, in
    the same transaction.
X12 Chunk.salient SHALL be written only for chunks present in the digest's
    walked set with a nonempty top-term list; a chunk never walked SHALL NOT
    acquire a salient property from this writer.
X13 CommunitySummary SHALL be keyed on `cid` alone (one mirror per run), and
    TOUCHED SHALL be re-writable: a second write of the same walk updates
    `hits` in place rather than duplicating the relationship.
X14 PATHWAY.ppr SHALL carry W20's `ppr` beside `dwpc` as an ADDITIVE property: dwpc
    still orders the pairs and keeps its name, and a caller whose pairs predate
    W20 writes null, which removes the property rather than freezing a stale
    number in the mirror.
X15 Community density and conductance (W19) SHALL be written to the
    CommunitySummary NODE, not to TOUCHED: `hits` is a property of one walk,
    density and conductance are properties of the stored partition, and putting
    them on the relationship would mint one copy per walk that could disagree.
    A caller that supplies no metrics writes null, and re-write semantics (X13)
    are unchanged.

Usage:
    python export_neo4j.py brown-50-dual out/          # both edge kinds
    python export_neo4j.py brown-50-dual out/ --edges contains
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import graph_tools as gt

MIN_TF = 1          # emit a CONTAINS edge at this term frequency or above

NEO4J_HTTP = "http://localhost:7474"        # docker compose maps the HTTP endpoint
NEO4J_DB = "neo4j"
NEO4J_AUTH = ("neo4j", "graphgraph")
TX_BATCH = 500                              # rows per UNWIND statement


def _fetch(conn, run, min_tf: int):
    """Guarantee: (chunks, contains, similar) as plain row lists."""
    with conn.cursor() as cur:
        cur.execute(
            """SELECT ord, doc_id, body, attrs -> 'tf' AS tf,
                      attrs ->> 'source' AS source
                 FROM node WHERE run_id = %s ORDER BY ord""", (run.run_id,))
        rows = cur.fetchall()
        cur.execute(
            """SELECT src, dst, strength, provenance, sim_sparse, sim_dense
                 FROM edge WHERE run_id = %s AND valid_to IS NULL
                ORDER BY src, dst""", (run.run_id,))
        similar = cur.fetchall()
        cur.execute(
            """SELECT cid, unnest(members) AS ord
                 FROM community WHERE run_id = %s""", (run.run_id,))
        cid = {r["ord"]: r["cid"] for r in cur.fetchall()}
        cur.execute(
            """SELECT ord, embedding::text AS e FROM node_embedding
                WHERE run_id = %s""", (run.run_id,))
        emb = {r["ord"]: [tok.strip() for tok in r["e"].strip("[]").split(",")]
               for r in cur.fetchall()}

    chunks, contains = [], []
    for r in rows:
        chunks.append((r["ord"], r["doc_id"], r["source"], r["body"],
                        cid.get(r["ord"]), emb.get(r["ord"])))
        for term, tf in (r["tf"] or {}).items():
            if tf >= min_tf:
                contains.append((r["ord"], term, tf))
    return chunks, contains, similar


def export(conn, run, out: Path, edges: str = "both", min_tf: int = MIN_TF) -> dict:
    """Require: out is a writable directory. Guarantee: X1-X8 hold on the files."""
    out.mkdir(parents=True, exist_ok=True)
    chunks, contains, similar = _fetch(conn, run, min_tf)

    terms = sorted({t for _, t, _ in contains})                     # X2
    chunk_ids = {c[0] for c in chunks}
    src_mix = gt.source_mix({"source": c[2], "doc_id": c[1]} for c in chunks)
    term_ids = set(terms)

    dangling = ([e for e in contains if e[1] not in term_ids]       # X3
                + [e for e in similar
                   if e["src"] not in chunk_ids or e["dst"] not in chunk_ids])
    if dangling:
        raise ValueError(f"{len(dangling)} dangling edge endpoints; "
                         f"first: {dangling[0]}")

    embed_dim = None                                                 # X7
    for _, _, _, _, _, e in chunks:
        if e:
            embed_dim = len(e)
            break
    n_embedded = 0
    n_labeled = 0
    for ord_, _, _, _, c, e in chunks:
        if e:
            n_embedded += 1
            if len(e) != embed_dim:
                raise ValueError(f"ragged embedding at ord {ord_}: "
                                  f"{len(e)} != {embed_dim}")
        if c is not None:
            n_labeled += 1

    with (out / "chunks.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)                                            # X4, X6
        w.writerow(["id:ID(Chunk)", "doc_id", "source", "text", "cid:int",
                    "embedding:float[]", ":LABEL"])
        for ord_, doc_id, source, body, c, e in chunks:
            label = "Chunk" if c is None else f"Chunk;C{c}"           # X8
            w.writerow([ord_, doc_id, source or "", body,
                       "" if c is None else c,
                       "" if not e else ";".join(e), label])

    with (out / "terms.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name:ID(Term)", ":LABEL"])
        for t in terms:
            w.writerow([t, "Term"])

    n_contains = 0
    with (out / "contains.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([":START_ID(Chunk)", ":END_ID(Term)", ":TYPE", "tf:int"])
        if edges in ("both", "contains"):
            for ord_, term, tf in contains:
                w.writerow([ord_, term, "CONTAINS", tf])
                n_contains += 1

    n_similar = 0
    with (out / "similar.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([":START_ID(Chunk)", ":END_ID(Chunk)", ":TYPE", "strength:float",
                    "prov", "sim_sparse:float", "sim_dense:float"])
        if edges in ("both", "similar"):
            for e in similar:
                w.writerow([e["src"], e["dst"], "SIMILAR", e["strength"],
                            e["provenance"], e["sim_sparse"],
                            "" if e["sim_dense"] is None else e["sim_dense"]])
                n_similar += 1

    src_line = (f"# sources: {gt.format_source_mix(src_mix)}\n"
                if src_mix else "")                                  # X5's spirit
    vector_line = ("# then: cypher-shell -f vector_index.cypher\n"
                    if embed_dim else "")
    (out / "import.sh").write_text(
        f"#!/bin/sh\n"
        f"# run {run.label} ({run.run_id})\n"                        # X5
        f"{src_line}"
        f"neo4j-admin database import full \\\n"
        f"  --nodes=chunks.csv \\\n"
        f"  --nodes=terms.csv \\\n"
        f"  --relationships=contains.csv \\\n"
        f"  --relationships=similar.csv \\\n"
        f"  --id-type=STRING \\\n"
        f"  --multiline-fields=true \\\n"
        f"  --overwrite-destination=true \\\n"
        f"  neo4j\n"
        f"{vector_line}", encoding="utf-8")

    if embed_dim:                                                    # X7
        (out / "vector_index.cypher").write_text(
            f"// run {run.label} ({run.run_id})\n"
            f"CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS\n"
            f"FOR (c:Chunk) ON (c.embedding)\n"
            f"OPTIONS {{indexConfig: {{`vector.dimensions`: {embed_dim}, "
            f"`vector.similarity_function`: 'cosine'}}}};\n",
            encoding="utf-8")
    else:
        cyp = out / "vector_index.cypher"
        if cyp.exists():
            cyp.unlink()

    return {"chunks": len(chunks), "terms": len(terms),
            "contains": n_contains, "similar": n_similar,
            "sources": src_mix,
            "embeddings": n_embedded, "embed_dim": embed_dim,
            "labeled": n_labeled,
            "label": run.label, "run_id": str(run.run_id)}


def _tx(statements: list, url: str = NEO4J_HTTP, auth: tuple = NEO4J_AUTH,
        db: str = NEO4J_DB, timeout: float = 30.0) -> list:
    """Require: a reachable neo4j HTTP tx endpoint.
    Guarantee: every statement committed, or an exception. Returns results."""
    body = json.dumps({"statements": statements}).encode("utf-8")
    token = base64.b64encode(f"{auth[0]}:{auth[1]}".encode("utf-8")).decode("ascii")
    req = urllib.request.Request(
        f"{url}/db/{db}/tx/commit", data=body,
        headers={"Content-Type": "application/json",
                 "Accept": "application/json",
                 "Authorization": f"Basic {token}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.load(r)
    errors = payload.get("errors") or []
    if errors:                                                      # fail fast
        first = errors[0]
        raise RuntimeError(f"neo4j tx error {first.get('code')}: {first.get('message')}")
    return payload["results"]


def write_walk(bundle, paths: dict, *, prompt: str | None = None,
               url: str = NEO4J_HTTP, auth: tuple = NEO4J_AUTH,
               db: str = NEO4J_DB, batch: int = TX_BATCH,
               post=_tx) -> dict:
    """Require: the run's chunks are already imported (id:ID(Chunk), STRING space).
    Guarantee: exactly one (:Walk {prompt}) node, its ANCHORS set, and one
               PATHWAY per pair per `of`, idempotent under re-write.
    Maintain:  no node creation in the Chunk id-space; chunks are MATCHed only."""
    prompt = prompt if prompt is not None else bundle.query

    anchors = []
    seen = set()
    for a in bundle.anchors:                                        # X9
        aid = str(a["ord"]) if isinstance(a, dict) else str(int(a))
        if aid not in seen:
            seen.add(aid)
            anchors.append(aid)

    rows = []
    for pair in paths.get("pairs", []):
        a, b = int(pair["a"]), int(pair["b"])
        lo, hi = (a, b) if a <= b else (b, a)                        # X10 normalize
        rows.append({
            "a": str(lo), "b": str(hi),
            "dwpc": pair["dwpc"], "n_paths": pair["n_paths"],
            "len": (len(pair["path"]) - 1) if pair.get("path") else None,
            "ppr": pair.get("ppr"),                                   # X14
        })

    # Schema modification cannot share a transaction with writes (neo4j rejects
    # it: "Write query after executing Schema modification"), so the constraint
    # goes over the wire on its own, before the write transaction.
    post([{
        "statement": "CREATE CONSTRAINT walk_prompt IF NOT EXISTS "
                      "FOR (w:Walk) REQUIRE w.prompt IS UNIQUE",
        "parameters": {},
    }], url=url, auth=auth, db=db)

    statements = [{
        "statement": ("MERGE (w:Walk {prompt: $prompt})\n"
                      "SET w.run_id = $run_id, w.n = $n, w.edges = $edges, "
                      "w.wcc = $wcc, w.wcc_sizes = $wcc_sizes, "
                      "w.largest_component_frac = $lcf, w.density = $density, "
                      "w.conductance = $conductance"),
        "parameters": {
            "prompt": prompt, "run_id": getattr(bundle, "run_id", None),
            "n": paths["n"], "edges": paths["edges"],
            "wcc": paths["components"], "wcc_sizes": paths["wcc_sizes"],
            "lcf": paths["largest_component_frac"],
            "density": paths["density"], "conductance": paths["conductance"],
        },
    }]

    anchor_stmt_idx = []
    for i in range(0, len(anchors), batch):
        anchor_stmt_idx.append(len(statements))
        statements.append({
            "statement": ("MATCH (w:Walk {prompt: $prompt})\n"
                          "UNWIND $ids AS cid\n"
                          "MATCH (c:Chunk {id: cid})\n"                # X9: MATCH, never MERGE
                          "MERGE (w)-[:ANCHORS]->(c)\n"
                          "RETURN count(c) AS matched"),
            "parameters": {"prompt": prompt, "ids": anchors[i:i + batch]},
        })

    pathway_stmt_idx = []
    for i in range(0, len(rows), batch):
        pathway_stmt_idx.append(len(statements))
        statements.append({
            "statement": ("UNWIND $rows AS r\n"
                          "MATCH (s:Chunk {id: r.a}), (d:Chunk {id: r.b})\n"
                          "MERGE (s)-[p:PATHWAY {of: $prompt}]->(d)\n"
                          "SET p.dwpc = r.dwpc, p.n_paths = r.n_paths, p.len = r.len, p.ppr = r.ppr\n"
                          "RETURN count(p) AS merged"),
            "parameters": {"prompt": prompt, "rows": rows[i:i + batch]},
        })

    results = post(statements, url=url, auth=auth, db=db)

    matched = 0
    for idx in anchor_stmt_idx:
        matched += results[idx]["data"][0]["row"][0]
    if matched != len(anchors):                                     # X10 shortfall guard
        missing = [a for a in anchors][:1]
        raise ValueError(f"anchors: matched {matched} of {len(anchors)}; "
                         f"first expected id {missing[0] if missing else None}")

    merged = 0
    for idx in pathway_stmt_idx:
        merged += results[idx]["data"][0]["row"][0]
    if merged != len(rows):
        missing = rows[0] if rows else None
        raise ValueError(f"pathways: merged {merged} of {len(rows)}; "
                         f"first expected pair {missing}")

    return {"prompt": prompt, "anchors": len(anchors), "pairs": len(rows),
            "statements": len(statements),
            **{k: paths[k] for k in
               ("n", "edges", "components", "density", "conductance")}}


def write_digest(bundle, digest: dict, touched: list, *, prompt: str | None = None,
                  url: str = NEO4J_HTTP, auth: tuple = NEO4J_AUTH,
                  db: str = NEO4J_DB, batch: int = TX_BATCH,
                  post=_tx) -> dict:
    """Require: an existing (:Walk {prompt}) node (write_walk already ran).
    Guarantee: NEXT_IN_CHAIN edges keyed on (src, dst, of, chain, pos) with
               stale of-scoped edges deleted first (X11); Chunk.salient set
               only for walked chunks with a nonempty top list (X12);
               CommunitySummary keyed on cid with TOUCHED re-writable (X13).
    Maintain:  no node creation in the Chunk id-space; chunks are MATCHed only
               (X9). Any transport or count-shortfall raises; no partial-write
               fallback."""
    prompt = prompt if prompt is not None else bundle.query

    chain_rows = []
    for ci, chain in enumerate(digest.get("chunks", {}).get("chains", [])):
        for pos, (a, b) in enumerate(zip(chain, chain[1:])):
            chain_rows.append({"src": str(a), "dst": str(b), "chain": ci, "pos": pos})

    sal = digest.get("sal", {})
    kept = digest.get("kept", list(sal.keys()))
    sal_rows = [{"id": str(o), "top": sal[o]["top"]}
                for o in kept if o in sal and sal[o].get("top")]

    touched_rows = [{"cid": t["cid"], "keywords": t["keywords"],
                      "size": t["size"], "hits": t["hits"],
                      "density": t.get("density"), "conductance": t.get("conductance")}
                     for t in touched]                                        # X15

    # Schema modification cannot share a transaction with writes (same
    # constraint as write_walk's walk_prompt).
    post([{
        "statement": "CREATE CONSTRAINT community_summary_cid IF NOT EXISTS "
                      "FOR (cs:CommunitySummary) REQUIRE cs.cid IS UNIQUE",
        "parameters": {},
    }], url=url, auth=auth, db=db)

    statements = [{
        "statement": "MATCH ()-[c:NEXT_IN_CHAIN {of: $prompt}]->() DELETE c",  # X11
        "parameters": {"prompt": prompt},
    }]

    chain_stmt_idx = []
    for i in range(0, len(chain_rows), batch):
        chain_stmt_idx.append(len(statements))
        statements.append({
            "statement": ("UNWIND $rows AS r\n"
                          "MATCH (s:Chunk {id: r.src}), (d:Chunk {id: r.dst})\n"    # X9
                          "MERGE (s)-[c:NEXT_IN_CHAIN {of: $prompt, chain: r.chain, "
                          "pos: r.pos}]->(d)\n"
                          "RETURN count(c) AS merged"),
            "parameters": {"prompt": prompt, "rows": chain_rows[i:i + batch]},
        })

    sal_stmt_idx = []
    for i in range(0, len(sal_rows), batch):
        sal_stmt_idx.append(len(statements))
        statements.append({
            "statement": ("UNWIND $rows AS r\n"
                          "MATCH (c:Chunk {id: r.id})\n"                          # X9, X12
                          "SET c.salient = r.top\n"
                          "RETURN count(c) AS matched"),
            "parameters": {"rows": sal_rows[i:i + batch]},
        })

    touched_stmt_idx = []
    for i in range(0, len(touched_rows), batch):
        touched_stmt_idx.append(len(statements))
        statements.append({
            "statement": ("UNWIND $rows AS r\n"
                          "MERGE (cs:CommunitySummary {cid: r.cid})\n"             # X13
                          "SET cs.keywords = r.keywords, cs.size = r.size, "
                          "cs.density = r.density, cs.conductance = r.conductance\n"
                          "WITH cs, r\n"
                          "MATCH (w:Walk {prompt: $prompt})\n"
                          "MERGE (w)-[t:TOUCHED]->(cs)\n"
                          "SET t.hits = r.hits\n"
                          "RETURN count(t) AS merged"),
            "parameters": {"prompt": prompt, "rows": touched_rows[i:i + batch]},
        })

    results = post(statements, url=url, auth=auth, db=db)

    merged_chains = sum(results[idx]["data"][0]["row"][0] for idx in chain_stmt_idx)
    if merged_chains != len(chain_rows):
        raise ValueError(f"chains: merged {merged_chains} of {len(chain_rows)}")

    matched_sal = sum(results[idx]["data"][0]["row"][0] for idx in sal_stmt_idx)
    if matched_sal != len(sal_rows):
        raise ValueError(f"salient: matched {matched_sal} of {len(sal_rows)}")

    merged_touched = sum(results[idx]["data"][0]["row"][0] for idx in touched_stmt_idx)
    if merged_touched != len(touched_rows):
        raise ValueError(f"touched: merged {merged_touched} of {len(touched_rows)}")

    return {"prompt": prompt, "chains": len(chain_rows), "salient": len(sal_rows),
            "touched": len(touched_rows), "statements": len(statements)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("label")
    ap.add_argument("out", type=Path)
    ap.add_argument("--edges", choices=("both", "contains", "similar"),
                    default="both",
                    help="contains = term-mediated only; drops the dense space")
    ap.add_argument("--min-tf", type=int, default=MIN_TF)
    a = ap.parse_args(argv)

    with gt.connect() as conn:
        run = gt.get_run(conn, a.label)
        stats = export(conn, run, a.out, a.edges, a.min_tf)
    for k, v in stats.items():
        print(f"{k:10s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
