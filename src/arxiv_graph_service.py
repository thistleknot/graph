"""arxiv_graph_service.py -- keep the arxiv graph current: newly extracted markdowns in, a rebuild when enough have arrived.

NO GOVERNING SPEC beyond operator instruction 2026-10-03: "setup a service just like arxiv-llmtxt-ingest to
automagically process remaining extracted markdowns into this graph representation for us, checking every 30
minutes (15 minutes after arxiv-llmtxt-ingest)". Tasks: playbook.md T123, T124. Skill: sparsevec-lexsem-graph.

    STAGE     MECHANISM                                                                          GUARD
    WATCH     one loop, a cycle every 30 minutes on a fixed minute of the hour                    S7
    GUARD     skip the cycle: hold file, VRAM above half used, another cycle running              S5
    FIND      post_processed/*.md whose doc_id lex_chunk_meta does not hold                       S2
    CHUNK     domain_corpora.chunk_arxiv with the live build's FROZEN fit; identical texts once   S1, S2
    FLAG      is_reference, is_junk (C9, C10): stored, never dropped
    VECTORS   sparse ip + cosine rows from ingest.doc_row under frozen statistics; dense MiniLM    S1, S3
              centered on the build's frozen mean
    PLACE     nearest unit centroid -> lex_assign(how = 'nearest-centroid', no layout position)    S4
    COMMIT    finish embeddings first; replace metadata, vectors, placement and mentions atomically S11
    REBUILD   chunks added since the build reach REBUILD_FRACTION of its size -> full pipeline     S6

S1  A build freezes its vocabulary, idf, avgdl, whale columns, chunking fit and embedding mean
    (lex_build.params). An incremental ingest reads them and changes none, so a row added now equals the
    row a full build would give that document (tests/test_ingest_arxiv_sparsevec.py pins doc_row).
    Terms the vocabulary has never seen carry no weight until the next full build.
S2  A document is ingested once (its doc_id is in lex_chunk_meta -> skipped) and a chunk whose text is
    already stored is skipped (chunker guard C8, across runs).
S8  A paper is its arXiv id without the version, plus `_methods` for the methods extract (domain_corpora
    C11); the version is a secondary key, 1 when the file name has none. A file replaces what is stored only
    with a HIGHER version: the old chunks are deleted first (before the duplicate-text check, since a new
    version's chunks mostly equal the old ones), then the new ones appended; the same or a lower version is
    not read. Overwritten chunks count toward the rebuild trigger like added ones.
S3  Reference and junk chunks are stored and flagged. A junk chunk gets no vectors at all; a reference
    chunk gets the sparse rows (A3 of the ingest) but no dense vector and no community.
S4  Placement is provisional. A placed chunk is marked `nearest-centroid`, has no UMAP position, and
    counts toward the rebuild trigger; the rebuild re-derives every label from consensus.
S5  One cycle at a time: a lock file holding a live pid. The VRAM rule is arxiv-llmtxt-ingest's own:
    above half of VRAM in use the cycle is deferred to the next slot, never run partially.
S6  REBUILD_FRACTION = 0.05 is a STARTING CONSTANT, not a derived one; every cycle logs the fraction so
    it can be moved. A rebuild runs chunk -> ingest -> map -> summarize -> render, aborts loudly at the
    first failing step, and a later cycle resumes from the first step the live build is missing
    (pending_steps). Known limitation: ingest replaces the label's vector tables in place, so
    retrieval over them is unavailable for the minutes a rebuild takes.
S7  The sibling fires every 15 minutes (WATCH_INTERVAL_S = 900) and its minute drifts by each cycle's
    duration. Slots are 30 minutes apart at (sibling minute mod 15) + 15 + GRACE_MIN, re-read from the
    sibling's log before every wait, so this service follows its drift and always starts two minutes
    after one of its cycles (a 15-minute sibling has no unique "15 minutes after": +15 lands on its
    next cycle, +17 on two minutes after it).

S9  Entities are opt-in per build. `--freeze-entities` derives an inventory from the live build's retrievable chunks
    (entity_derive, guards AE1-AE14), stores it in lex_entity, records the tokenizer version and the parameters in
    lex_build.params["entities"], and matches every retrievable chunk. A cycle then matches each newly ingested
    retrievable chunk against that frozen inventory and changes no statistic (AE11). A build with no inventory does
    nothing here; a full rebuild creates a new build and, only if the one it replaces had an inventory, re-freezes
    with the same parameters. An inventory whose tokenizer version differs from the code's is never matched against.
    Task T131; spec: the approved plan derive-arxiv-entities-with-information-theory.md.

S10 Before a cycle touches Postgres, ensure_docker starts Docker Desktop when the engine does not answer and
    `docker compose up -d graphdb` when chunkgraph-pg is not healthy, waiting up to DOCKER_WAIT_S for each.
    Still down -> the cycle is skipped and its papers stay pending for the next slot.
S11 Embeddings and derived placement are computed before database mutation. Replacement deletion, chunk metadata,
    sparse vectors, dense vectors, assignments and entity mentions commit as one transaction; a failure rolls back.
    A stored paper missing a dense row or assignment is reloaded from its current markdown and replaced on retry.
S12 `--root` selects the Markdown `papers/post_processed` directory for one-shot and watch modes; the default is
    `domain_corpora.ARXIV_DIR`.
S13 Each ingest cycle ensures the legacy `lex_build.ready` column exists before any live-build read; older graph
    database schemas may predate the generation readiness flag.

Run:  python -u src\\arxiv_graph_service.py --watch | --once | --rebuild | --freeze-entities
"""
from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import hashlib
import json
import os
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

import entity_derive as ed
import sparsevec_store as ss
from domain_corpora import _read, arxiv_files, chunk_arxiv, load_arxiv_papers, with_junk_flag
from ingest_arxiv_sparsevec import doc_row
from stoplist import tokenize

LABEL = "arxiv_sect"
CACHE = ROOT / ".tmp" / "arxiv_section_chunks.pkl"
MAP_DIR = ROOT / ".tmp"             # where src/arxiv_community_map.py writes its PNG
HOLD, LOCK = ROOT / ".tmp" / "arxiv_graph_hold", ROOT / ".tmp" / "arxiv_graph.lock"
SIBLING_LOG = Path(r"D:\c\arxiv_id_lists\.tmp\watch.log")
ENTITY_PARAMS = ed.Params(drop=frozenset({"bh"}), df_floor=2, uni_quota=None)    # S9: chosen by the T132 sweep's pre-registered rule (two disjoint paper samples, tuning half); held-out: matches 1.36 -> 0.14 per no-entity sentence, recall 14/43 -> 11/43 (McNemar p=0.45)
ENTITY_K = 40_000                   # S9: inventory size for the 2,521-paper corpus; pre-registered rule over k in {5k..80k}, tuning half: recall 7/41 at 5k, 26/41 at 40k and 80k, matches per no-entity sentence 0.92 at 40k vs 2.69 at 80k. k does NOT carry over to a corpus of another size.
REBUILD_FRACTION = 0.05             # S6: a starting constant
VRAM_DEFER_FRAC = 0.5               # S5: the sibling's own rule
INTERVAL_MIN, SIBLING_PERIOD_MIN, OFFSET_MIN, GRACE_MIN = 30, 15, 15, 2
STEPS = ("chunk", "ingest", "map", "summarize", "render")
IDLE_PRIORITY_CLASS = 0x00000040
DOCKER_DESKTOP = r"C:\Program Files\Docker\Docker\Docker Desktop.exe"   # S10
PG_CONTAINER = "chunkgraph-pg"      # S10: docker-compose.yml service graphdb
DOCKER_WAIT_S = 180.0               # S10: engine start and container health, each


def log(msg: str) -> None:
    print("[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg), flush=True)


# ------------------------------------------------------------------ schedule ----
def sibling_minute(path: Path = SIBLING_LOG) -> int | None:
    """S7. Guarantee: the minute-of-hour of the sibling's newest `[watch ]` line, or None when the log is
    missing or has none. The log is written by PowerShell as UTF-16, so both encodings are read, from the
    tail only."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as fh:
            fh.seek(max(0, size - 200_000) // 2 * 2)
            raw = fh.read()
    except OSError:
        return None
    text = raw.decode("utf-16-le", "ignore") if b"\x00" in raw[:400] else raw.decode("utf-8", "ignore")
    stamps = re.findall(r"\[watch \] \d{4}-\d\d-\d\d \d\d:(\d\d)", text)
    return int(stamps[-1]) if stamps else None


def next_slot(now: dt.datetime, sib_minute: int | None) -> dt.datetime:
    """S7. Guarantee: the earliest whole minute strictly after `now` whose minute-of-hour is congruent, mod
    INTERVAL_MIN, to (sib_minute mod SIBLING_PERIOD_MIN) + OFFSET_MIN + GRACE_MIN (sibling unknown -> 0)."""
    phase = ((sib_minute % SIBLING_PERIOD_MIN if sib_minute is not None else 0) + OFFSET_MIN + GRACE_MIN) % INTERVAL_MIN
    t = now.replace(second=0, microsecond=0)
    for _ in range(INTERVAL_MIN + 1):
        t += dt.timedelta(minutes=1)
        if t > now and t.minute % INTERVAL_MIN == phase:
            return t
    raise AssertionError("no slot found")


# ---------------------------------------------------------------------- guard ----
def vram_used_frac() -> float | None:
    """S5. Guarantee: fraction of VRAM in use on the busiest GPU, None when nvidia-smi is unavailable."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        return max(int(u) / int(t) for u, t in (ln.split(",") for ln in out)) if out else None
    except Exception:
        return None


def ensure_docker(run=subprocess.run, start=subprocess.Popen, sleep=time.sleep,
                  wait_s: float = DOCKER_WAIT_S) -> bool:
    """S10. Guarantee: True once the Docker engine answers and PG_CONTAINER reports healthy. An engine that
    does not answer gets Docker Desktop started, a container that is not up gets `docker compose up -d graphdb`
    (ROOT/docker-compose.yml), and both are polled up to `wait_s`. False when either never comes up: the cycle
    is skipped and the pending papers are retried next slot. Operator, 2026-10-20: Docker Desktop had to be
    started by hand before the arxiv papers reached Postgres."""
    def ok(*cmd) -> str | None:
        try:
            r = run(list(cmd), capture_output=True, text=True, timeout=90)   # `docker info` measured 22s, and >30s once, on a healthy engine
        except Exception:
            return None
        return r.stdout.strip() if r.returncode == 0 else None

    def poll(check) -> bool:
        deadline = time.monotonic() + wait_s
        while True:
            if check():
                return True
            if time.monotonic() >= deadline:
                return False
            sleep(5)

    if ok("docker", "info", "--format", "{{.ServerVersion}}") is None:
        log("docker engine down: starting Docker Desktop")
        start([DOCKER_DESKTOP])
        if not poll(lambda: ok("docker", "info", "--format", "{{.ServerVersion}}") is not None):
            log("docker engine still down after %.0fs: skipping cycle" % wait_s)
            return False
    health = lambda: ok("docker", "inspect", "-f", "{{.State.Health.Status}}", PG_CONTAINER)
    if health() != "healthy":
        log("%s not healthy: docker compose up -d graphdb" % PG_CONTAINER)
        ok("docker", "compose", "-f", str(ROOT / "docker-compose.yml"), "up", "-d", "graphdb")
        if not poll(lambda: health() == "healthy"):
            log("%s still not healthy after %.0fs: skipping cycle" % (PG_CONTAINER, wait_s))
            return False
    return True


def acquire_lock(path: Path = LOCK) -> bool:
    """S5. Guarantee: True and the lock file holds this pid, unless a LIVE process already holds it. A
    pid that no longer exists is a stale lock and is taken over. (os.kill(pid, 0) would terminate on
    Windows, so liveness is psutil's.)"""
    import psutil
    if path.exists():
        try:
            held = int(path.read_text().strip())
        except ValueError:
            held = -1
        if held > 0 and held != os.getpid() and psutil.pid_exists(held):
            return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(os.getpid()))
    return True


def release_lock(path: Path = LOCK) -> None:
    if path.exists() and path.read_text().strip() == str(os.getpid()):
        path.unlink()


# --------------------------------------------------------------------- ingest ----
def frozen(conn, label: str = LABEL) -> dict:
    """S1. Guarantee: everything an incremental ingest needs, read from the label's live build: ids, the
    chunking fit, BM25 statistics, vocabulary (col is 0-based), idf, dense mean, unit centroids. The build
    must be complete through the map stage, else this fails: nothing to freeze, nothing to place into."""
    live = ss.live_build(conn, label)
    assert live is not None, "no live build for %r: run a rebuild" % label
    build, n_chunks, p = live
    assert "dense" in p and "communities" in p, "build %d has no dense mean or communities: the map stage has not run" % build
    rows = conn.execute("SELECT col, piece, idf FROM lex_vocab WHERE label = %s ORDER BY col", (label,)).fetchall()
    cents = conn.execute("SELECT cid, centroid FROM lex_community WHERE build_id = %s AND centroid IS NOT NULL ORDER BY cid",
                         (build,)).fetchall()
    assert rows and cents, "build %d has an empty vocabulary or no centroids" % build
    return {"build": build, "n_chunks": n_chunks, "fit": p["fit"], "dim": p["bm25"]["dim"], "avgdl": p["bm25"]["avgdl"],
            "whales": p["bm25"]["whale_cols"], "col": {piece: c - 1 for c, piece, _ in rows},
            "idf": np.array([r[2] for r in rows], float), "mean": np.array(p["dense"]["mean"], np.float32),
            "entities": p.get("entities"), "dense_dim": p["dense"]["dim"], "cids": [c for c, _ in cents],
            "centroids": np.array([json.loads(v) for _, v in cents], np.float32)}


def frozen_inventory(conn, build: int, meta: dict | None) -> "ed.Inventory | None":
    """S9. `meta` is lex_build.params["entities"]. Guarantee: the build's frozen inventory as an entity_derive.Inventory, or None
    when none was frozen or its tokenizer version is not the code's (a different tokenizer would match different spans, AE11)."""
    rows = ss.read_entities(conn, build) if meta else []
    if not rows:
        return None
    if meta.get("tokenizer") != ed.TOKENIZER_VERSION:
        log("inventory of build %d was frozen with tokenizer %r, code is %r: not matching" % (build, meta.get("tokenizer"), ed.TOKENIZER_VERSION))
        return None
    return ed.Inventory([r[1] for r in rows], [r[2] for r in rows], np.array([r[3] for r in rows], np.int64),
                        np.array([r[4] for r in rows], float), np.array([r[5] for r in rows], np.int64),
                        np.array([r[6] for r in rows], np.int64), np.array([r[7] for r in rows], np.int64),
                        np.array([r[8] for r in rows], np.int64), meta["tokenizer"])


def match_entities(conn, build: int, meta: dict | None, ords: list[int], texts: list[str], doc_ids: list[str]) -> int:
    """S9, AE11, AE13. Guarantee: the frozen inventory's mentions of the chunks `ords` are in lex_mention (their old ones
    replaced); returns rows written, 0 when the build has no usable inventory. No statistic is read or changed."""
    inv = frozen_inventory(conn, build, meta)
    if inv is None or not ords:
        return 0
    chunk, ent, cnt = ed.match_frozen(inv, ed.tokenize_corpus(texts, doc_ids))
    return ss.replace_mentions(conn, build, ords, [(ords[int(c)], int(e), int(n)) for c, e, n in zip(chunk, ent, cnt)])


def freeze_entities(conn, label: str = LABEL, params: "ed.Params | None" = None, k: int = ENTITY_K, batch: int = 4000) -> dict:
    """S9. Guarantee: an inventory derived from the live build's retrievable chunks is stored in lex_entity with its parameters in
    lex_build.params["entities"], and every retrievable chunk is matched against it. Replaces any inventory the build had."""
    import dataclasses
    params = params or ENTITY_PARAMS
    live = ss.live_build(conn, label)
    assert live is not None, "no live build for %r" % label
    build = live[0]
    ss.ensure_entity_schema(conn)
    rows = conn.execute("SELECT ord, doc_id, text FROM lex_chunk_meta WHERE label = %s AND NOT is_reference AND NOT is_junk ORDER BY ord",
                        (label,)).fetchall()
    ords, doc_ids, texts = [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]
    log("deriving entities from %d chunks of build %d" % (len(texts), build))
    inv, _, _, corpus, flags = ed.derive(texts, doc_ids, params, k=k)
    ss.write_entities(conn, build, [(i, inv.fold[i], inv.surface[i], int(inv.n[i]), float(inv.score[i]), int(inv.df_papers[i]),
                                     int(inv.df_chunks[i]), int(inv.tf[i]), int(inv.canonical[i])) for i in range(len(inv.fold))])
    meta = {"tokenizer": ed.TOKENIZER_VERSION, "k": k, "n": len(inv.fold), "params": {**dataclasses.asdict(params), "drop": sorted(params.drop)},
            "cipher_papers": [corpus.papers[i] for i in np.flatnonzero(flags)]}
    ss.update_build_params(conn, build, {"entities": meta})
    written = 0
    for i in range(0, len(texts), batch):
        written += match_entities(conn, build, meta, ords[i:i + batch], texts[i:i + batch], doc_ids[i:i + batch])
    log("froze %d entities (cipher papers left out: %d), %d mention rows" % (len(inv.fold), len(meta["cipher_papers"]), written))
    return {"entities": len(inv.fold), "mentions": written, "cipher_papers": len(meta["cipher_papers"])}


def ingest_new(conn, label: str, ids: list[str], docs: list[str], st: dict, embed_fn,
               versions: dict | None = None, replace: set | frozenset = frozenset()) -> dict:
    """Add new papers or atomically repair/replace papers in the frozen chunk graph.

    Require:
        `st` describes a complete live build, and each document ID matches its input text.
    Guarantee:
        A successful call commits metadata, sparse and dense vectors, assignments, and mentions together.
    Maintain:
        Replaced rows remain intact until embeddings are ready; a write failure rolls the replacement back.
    Assert:
        Every retrievable inserted chunk has a dense vector and community assignment.

    Args:
        conn: Open autocommit Postgres connection.
        label: Target sparsevec label.
        ids: Paper document IDs matching `docs`.
        docs: Latest Markdown texts to chunk.
        st: Frozen statistics and community data for the live build.
        embed_fn: Dense encoder for retrievable chunks.
        versions: Optional source version per document ID.
        replace: Existing document IDs to replace within the transaction.

    Returns:
        Counts of added, duplicate, junk, reference, placed, and replaced chunks.

    Called by:
        `cycle` and deterministic tests.

    Depends on:
        `chunk_arxiv`, the frozen encoder contract, and `sparsevec_store` writes.

    Failure modes:
        Encoding errors precede writes; SQL and assertion errors roll back the transaction.
    """
    import hnsw_communities as hc
    versions = versions or {}
    records, _, stats = chunk_arxiv(ids, docs, fit=st["fit"])
    seen = ss.known_text_md5(conn, label, [r["text"] for r in records], sorted(replace))
    fresh = [r for r in with_junk_flag(records) if hashlib.md5(r["text"].encode("utf-8")).hexdigest() not in seen]
    for r in fresh:
        r["version"] = versions.get(r["doc_id"], 1)
    out = {"docs": len(ids), "chunks": len(fresh), "duplicates": len(records) - len(fresh) + stats["duplicates"],
           "junk": sum(r["is_junk"] for r in fresh), "references": sum(r["is_reference"] for r in fresh), "placed": 0,
           "replaced_chunks": 0}
    if not fresh and not replace:
        return out
    sat_rows, cos_rows = [], []
    for i, r in enumerate(fresh):
        if r["is_junk"]:
            continue
        sat, cos = doc_row(tokenize(r["text"]), st["col"], st["idf"], st["avgdl"], st["whales"])
        sat_rows.append((i, r["doc_id"], "arxiv", sat))
        cos_rows.append((i, r["doc_id"], "arxiv", cos))
    keep = [i for i, r in enumerate(fresh) if not r["is_reference"] and not r["is_junk"]]
    E = None
    nearest = None
    if keep:
        E = hc.center(embed_fn([fresh[i]["text"] for i in keep]), st["mean"])
        assert E.shape[1] == st["dense_dim"], "embedding dimension %d, build has %d" % (E.shape[1], st["dense_dim"])
        nearest = np.argmax(E @ st["centroids"].T, axis=1)
    with conn.transaction():
        removed = ss.delete_papers(conn, label, sorted(replace), st["build"])
        first = ss.append_chunk_meta(conn, label, fresh, min_ord=st["n_chunks"])
        ords = [first + i for i in range(len(fresh))]
        sat_rows = [(ords[i], doc_id, source, vec) for i, doc_id, source, vec in sat_rows]
        cos_rows = [(ords[i], doc_id, source, vec) for i, doc_id, source, vec in cos_rows]
        ss.write_chunks(conn, label, st["dim"], sat_rows, replace=False)
        ss.write_chunks(conn, label, st["dim"], cos_rows, replace=False, kind="cos")
        if keep:
            dense_ords = [ords[i] for i in keep]
            ss.write_dense(conn, label, dense_ords, E, replace=False)
            ss.write_assignments(conn, st["build"], [(ords[i], int(st["cids"][c]), "nearest-centroid", None, None)
                                                     for i, c in zip(keep, nearest)])
            out["placed"] = len(keep)
            if st.get("entities"):
                out["mentions"] = match_entities(conn, st["build"], st["entities"], dense_ords,
                                                 [fresh[i]["text"] for i in keep], [fresh[i]["doc_id"] for i in keep])
        assert out["placed"] == len(keep), "not every retrievable chunk was assigned"
    out["replaced_chunks"] = removed
    return out


def incomplete_papers(conn, label: str, build_id: int) -> list[str]:
    """Find stored papers that cannot participate in dense/community retrieval.

    Args:
        conn: Open Postgres connection.
        label: Sparsevec label to inspect.
        build_id: Live community build ID.

    Returns:
        Sorted document IDs with at least one retrievable chunk missing a dense row or assignment.

    Failure modes:
        Missing schema or database errors propagate.
    """
    dense = ss._table(label, "dense")
    rows = conn.execute(f"""SELECT m.doc_id
        FROM lex_chunk_meta m
        LEFT JOIN {dense} d ON d.ord = m.ord
        LEFT JOIN lex_assign a ON a.build_id = %s AND a.ord = m.ord
        WHERE m.label = %s AND NOT m.is_junk AND NOT m.is_reference
        GROUP BY m.doc_id
        HAVING count(*) FILTER (WHERE d.ord IS NOT NULL) < count(*)
            OR count(*) FILTER (WHERE a.ord IS NOT NULL) < count(*)
        ORDER BY m.doc_id""", (build_id, label)).fetchall()
    return [row[0] for row in rows]


def load_incomplete_papers(root: str | None, ids: list[str], stored: dict[str, int]) -> list[dict]:
    """Reload incomplete papers from the newest source file without downgrading stored versions.

    Args:
        root: Markdown corpus root, or None to use the repository default.
        ids: Incomplete arXiv document IDs.
        stored: Current document versions from the database.

    Returns:
        Latest source records at the stored version or higher, each marked for replacement.
    """
    wanted = set(ids)
    files = {"arxiv/" + key: (version, path) for key, version, path in arxiv_files(root) if "arxiv/" + key in wanted}
    out = []
    for doc_id in sorted(wanted):
        if doc_id not in files:
            continue
        version, path = files[doc_id]
        if version < stored.get(doc_id, 0):
            continue
        text = _read(path)
        if text.strip():
            out.append({"doc_id": doc_id, "version": version, "path": path, "text": text,
                        "upgrade": version > stored.get(doc_id, 0), "repair": True})
    return out


# -------------------------------------------------------------------- rebuild ----
def map_is_current(created: dt.datetime, png_dir: Path | None = None) -> bool:
    """S6. Guarantee: True when a community-map PNG, the stable name or a timestamped sibling the viewer
    fallback wrote, is at least as new as the build. A render that failed after a finished build leaves no
    other trace, so its absence has to be read off the file."""
    pngs = list((png_dir or MAP_DIR).glob("arxiv_community_map*.png"))
    return bool(pngs) and max(p.stat().st_mtime for p in pngs) >= created.timestamp()


def pending_steps(conn, label: str = LABEL, png_dir: Path | None = None) -> list[str]:
    """S6. Guarantee: the pipeline steps the label's live build still lacks, in order: all of them with no
    build; from `map` when the build has no communities; `summarize` and `render` when summaries are missing;
    `render` alone when everything is there but no map picture is as new as the build."""
    live = ss.live_build(conn, label)
    if live is None:
        return list(STEPS)
    build, _, p = live
    if "communities" not in p:
        return list(STEPS[2:])
    done = conn.execute("SELECT count(*) FROM lex_summary WHERE build_id = %s", (build,)).fetchone()[0]
    if done < p["communities"]["n"]:
        return list(STEPS[3:])
    created = conn.execute("SELECT created FROM lex_build WHERE build_id = %s", (build,)).fetchone()[0]
    return [] if map_is_current(created, png_dir) else ["render"]


def changed_fraction(conn, label: str, n_chunks: int) -> float:
    """S6. Guarantee: chunks added since the build plus chunks of the build that were overwritten away, over the
    build's size. Ords below n_chunks are the build's own; ords never get reused."""
    added = conn.execute("SELECT count(*) FROM lex_chunk_meta WHERE label = %s AND ord >= %s", (label, n_chunks)).fetchone()[0]
    kept = conn.execute("SELECT count(*) FROM lex_chunk_meta WHERE label = %s AND ord < %s", (label, n_chunks)).fetchone()[0]
    return round((added + (n_chunks - kept)) / n_chunks, 4)


def rebuild_chunks(root: str | None = None, out: Path = CACHE) -> int:
    """C1-C11. Guarantee: the latest version of every post_processed paper chunked with a fresh fit and cached
    for the ingest and map steps, each record carrying its paper's version. Returns the chunk count."""
    papers = load_arxiv_papers(root=root)
    ids, docs = [p["doc_id"] for p in papers], [p["text"] for p in papers]
    recs, fit, stats = chunk_arxiv(ids, docs)
    by_id = {p["doc_id"]: p["version"] for p in papers}
    for r in recs:
        r["version"] = by_id[r["doc_id"]]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump({"records": recs, "fit": fit, "stats": stats}, fh)
    log("chunked %d papers -> %d chunks, fit %s" % (len(ids), len(recs), fit))
    return len(recs)


def run_step(name: str, root: str | None = None) -> None:
    """S6. Fail loudly: a step that exits non-zero raises, and the rebuild stops there."""
    if name == "chunk":
        rebuild_chunks(root)
        return
    tool = {"ingest": ["src/ingest_arxiv_sparsevec.py", "--no-battery"], "map": ["src/arxiv_community_map.py"],
            "summarize": ["src/summarize_clusters.py", "--all"], "render": ["src/arxiv_community_map.py"]}[name]
    t0 = time.time()
    rc = subprocess.run([sys.executable, "-u", *tool], cwd=ROOT).returncode
    log("step %-9s exit %d  %.0fs" % (name, rc, time.time() - t0))
    if rc != 0:
        raise RuntimeError("rebuild step %r failed (exit %d)" % (name, rc))


def rebuild(steps: list[str] | None = None, root: str | None = None) -> None:
    for name in steps or list(STEPS):
        run_step(name, root)


# ---------------------------------------------------------------------- cycle ----
def cycle(label: str = LABEL, root: str | None = None, embed_fn=None, conn=None, rebuild_fn=None) -> dict:
    """S1-S6. One pass: finish an incomplete build, ingest new documents, rebuild if enough have arrived.
    Guarantee: returns what it did; nothing is changed for a corpus with no new document."""
    conn = conn or ss.connect()
    rebuild_fn = rebuild_fn or rebuild
    ss.ensure_build_schema(conn)                               # S13: older graph DBs predate generation readiness
    ss.ensure_chunk_meta(conn)                                   # columns a build older than this code lacks (V8, V9)
    fixed = ss.normalise_versioned_doc_ids(conn, label)          # a version left inside a doc_id would ingest the paper twice
    if fixed:
        log("moved the version out of the doc_id for %d stored chunks" % fixed)
    todo = pending_steps(conn, label)
    if todo:
        log("live build incomplete, resuming at %r" % todo[0])
        rebuild_fn(todo)
        return {"resumed": todo}
    st = frozen(conn, label)
    stored = ss.stored_versions(conn, label)
    repair_ids = incomplete_papers(conn, label, st["build"])
    papers = load_arxiv_papers(root=root, stored=stored)
    if repair_ids:
        repairs = load_incomplete_papers(root, repair_ids, stored)
        if len(repairs) < len(repair_ids):
            log("%d incomplete stored papers have no current source markdown at an equal or higher version" %
                (len(repair_ids) - len(repairs)))
        by_id = {paper["doc_id"]: paper for paper in papers}
        by_id.update({paper["doc_id"]: paper for paper in repairs})
        papers = [by_id[key] for key in sorted(by_id)]
    out = {"new_docs": sum(not p["upgrade"] for p in papers), "upgraded_docs": sum(p["upgrade"] for p in papers)}
    if papers:
        if embed_fn is None:
            from arxiv_community_map import encode_minilm as embed_fn
        out.update(ingest_new(conn, label, [p["doc_id"] for p in papers], [p["text"] for p in papers], st, embed_fn,
                              versions={p["doc_id"]: p["version"] for p in papers},
                              replace={p["doc_id"] for p in papers if p["upgrade"] or p.get("repair")}))
    out["added_fraction"] = changed_fraction(conn, label, st["n_chunks"])
    out["rebuild"] = out["added_fraction"] >= REBUILD_FRACTION
    log("cycle: %s" % json.dumps(out))
    if out["rebuild"]:
        log("changed %.1f%% of the build's chunks (threshold %.0f%%): full rebuild" % (100 * out["added_fraction"], 100 * REBUILD_FRACTION))
        prior = st.get("entities")                                # S9: only a build that had an inventory gets one again
        rebuild_fn(list(STEPS))
        if prior:
            out["entities"] = freeze_entities(conn, label, ed.Params(**{**prior["params"], "drop": frozenset(prior["params"].get("drop", []))}),
                                              k=prior["k"])
    return out


def watch(label: str = LABEL, root: str | None = None) -> None:
    """S5, S7, S12. Run forever on the sibling-relative schedule with an explicit corpus root when provided.

    Args:
        label: Sparsevec build label.
        root: Markdown root passed into every cycle.

    Failure modes:
        A cycle failure is logged and retried at the next slot.
    """
    try:
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), IDLE_PRIORITY_CLASS)
    except Exception:
        pass
    while True:
        slot = next_slot(dt.datetime.now(), sibling_minute())
        log("next cycle at %s" % slot.strftime("%H:%M"))
        while (left := (slot - dt.datetime.now()).total_seconds()) > 0:
            time.sleep(min(left, 30))
        used = vram_used_frac()
        if HOLD.exists():
            log("paused (%s present): skipping cycle" % HOLD.name)
        elif used is not None and used > VRAM_DEFER_FRAC:
            log("VRAM %.0f%% used > %.0f%%: GPU contention, deferring to the next slot" % (100 * used, 100 * VRAM_DEFER_FRAC))
        elif not acquire_lock():
            log("another cycle holds %s: skipping" % LOCK.name)
        elif not ensure_docker():
            release_lock()
        else:
            try:
                cycle(label, root=root)
            except Exception as exc:
                log("cycle crashed: %r" % exc)
            finally:
                release_lock()


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--watch", action="store_true", help="loop forever on the 30-minute slots")
    g.add_argument("--once", action="store_true", help="one cycle now")
    g.add_argument("--rebuild", action="store_true", help="the full pipeline now")
    g.add_argument("--freeze-entities", action="store_true", help="derive and store an entity inventory for the live build (S9)")
    ap.add_argument("--label", default=LABEL)
    ap.add_argument("--root", help="Markdown papers/post_processed directory (defaults to domain_corpora.ARXIV_DIR)")
    args = ap.parse_args()
    if args.watch:
        watch(args.label, root=args.root)
        return
    assert acquire_lock(), "another cycle holds %s" % LOCK
    try:
        assert ensure_docker(), "Postgres (%s) is not up" % PG_CONTAINER
        if args.freeze_entities:
            freeze_entities(ss.connect(), args.label)
        else:
            rebuild() if args.rebuild else cycle(args.label, root=args.root)
    finally:
        release_lock()


if __name__ == "__main__":
    main()
