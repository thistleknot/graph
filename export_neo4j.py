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
Guarantee: nodes.csv + edges.csv + import.sh in `out`, neo4j-admin header
           format, every :START_ID/:END_ID resolvable within its id-space.
           nodes.csv carries a `source` column, empty on runs predating R20
           (design.md §6.14 R20).
Maintain:  no writes; no re-partitioning; cid is read, never recomputed.

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

Usage:
    python export_neo4j.py brown-50-dual out/          # both edge kinds
    python export_neo4j.py brown-50-dual out/ --edges contains
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import graph_tools as gt

MIN_TF = 1          # emit a CONTAINS edge at this term frequency or above


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

    chunks, contains = [], []
    for r in rows:
        chunks.append((r["ord"], r["doc_id"], r["source"], r["body"], cid.get(r["ord"])))
        for term, tf in (r["tf"] or {}).items():
            if tf >= min_tf:
                contains.append((r["ord"], term, tf))
    return chunks, contains, similar


def export(conn, run, out: Path, edges: str = "both", min_tf: int = MIN_TF) -> dict:
    """Require: out is a writable directory. Guarantee: X1-X5 hold on the files."""
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

    with (out / "nodes.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)                                           # X4
        w.writerow(["id:ID(Chunk)", "id:ID(Term)", "doc_id", "source", "text",
                    "cid:int", ":LABEL"])
        for ord_, doc_id, source, body, c in chunks:
            w.writerow([ord_, "", doc_id, source or "", body,
                       "" if c is None else c, "Chunk"])
        for t in terms:
            w.writerow(["", t, "", "", "", "", "Term"])

    n_contains = n_similar = 0
    with (out / "edges.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([":START_ID(Chunk)", ":END_ID(Term)", ":END_ID(Chunk)",
                    ":TYPE", "tf:int", "strength:float", "prov",
                    "sim_sparse:float", "sim_dense:float"])
        if edges in ("both", "contains"):
            for ord_, term, tf in contains:
                w.writerow([ord_, term, "", "CONTAINS", tf, "", "", "", ""])
                n_contains += 1
        if edges in ("both", "similar"):
            for e in similar:
                w.writerow([e["src"], "", e["dst"], "SIMILAR", "",
                            e["strength"], e["provenance"],
                            e["sim_sparse"],
                            "" if e["sim_dense"] is None else e["sim_dense"]])
                n_similar += 1

    src_line = (f"# sources: {gt.format_source_mix(src_mix)}\n"
                if src_mix else "")                                  # X5's spirit
    (out / "import.sh").write_text(
        f"#!/bin/sh\n"
        f"# run {run.label} ({run.run_id})\n"                        # X5
        f"{src_line}"
        f"neo4j-admin database import full \\\n"
        f"  --nodes=nodes.csv \\\n"
        f"  --relationships=edges.csv \\\n"
        f"  --id-type=STRING \\\n"
        f"  --overwrite-destination=true \\\n"
        f"  neo4j\n", encoding="utf-8")

    return {"chunks": len(chunks), "terms": len(terms),
            "contains": n_contains, "similar": n_similar,
            "sources": src_mix,
            "label": run.label, "run_id": str(run.run_id)}


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
