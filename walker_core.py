"""walker_core.py -- the walker's UI-free logic: clip, labels, partitions, assess.

Spec: .spec/specs/graph-explorer/design.md 6.21(c) (walker split, UI wiring only)
Task: playbook.md T36

Imports NOTHING from streamlit, plotly or networkx and opens NO connection at
import time: `python -c "import walker_core"` must exit 0 with the database
down. Every function takes conn/run/state as parameters -- there is no module
state here. Spring layout deliberately stays in walker_app (design 6.21(a)).
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import evidence
import graph_tools as gt
import interpret


def clip(text: str, n: int) -> str:
    """Never cut inside a word (design 6.3). Clip at the last whitespace
    before n and mark the cut; short text is returned untouched."""
    text = text or ""
    if len(text) <= n:
        return text
    head = text[:n]
    cut = head.rsplit(None, 1)[0] if " " in head else head
    return cut + " …"


def load_labels(path: Path, run_id: str) -> dict:
    """Draft labels for THIS run, or {}. Absent file, unreadable file and
    malformed JSON are all the normal empty state.

    cid is RUN-LOCAL: a labels file drafted against another run is not stale,
    it is wrong -- it put 'early electrical science history' on the
    Moroccan elections. A foreign file returns {"__stale_run__": <its run_id>} so the
    caller can say so; it never returns that file's labels.
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if str(data.get("run_id")) != str(run_id):
        return {"__stale_run__": data.get("run_id")}
    return {c["cid"]: c for c in data.get("communities", []) if c.get("label")}


def src_of_map(conn, run, ords) -> dict:
    """{ord: source-or-None} for `ords`. One line, two call sites in the UI
    (the 3D tab and the Map partitions table) that each had their own copy."""
    return {o: gt.source_of(gt.node(conn, run, o)) for o in ords}


def partition_rows(ds: dict, src_of: dict) -> list[dict]:
    """One row per dendrite CHUNK chain: size, source mix, and the chain
    members' own BM25-salient vocabulary ranked by how many members carry it
    (top 12; 'term(n)' when n > 1). Pure: `ds` is dendrite_state's output,
    `src_of` comes from src_of_map. No DB, no st."""
    sal = ds["sal"]
    rows = []
    for ci, chain in enumerate(ds["chunks"]["chains"]):
        tc = Counter()
        mix = Counter()
        for o in chain:
            for t in (sal.get(o, {}).get("top") or []):
                tc[t] += 1
            mix[src_of.get(o) or "?"] += 1
        rows.append({
            "chain": ci + 1,
            "chunks": len(chain),
            "sources": " ".join(f"{k}:{v}" for k, v in mix.most_common()),
            "salient terms (carried by N members)":
                ", ".join(f"{t}({n})" if n > 1 else t
                          for t, n in tc.most_common(12)),
        })
    return rows


# ---------- analysis view (T39, design 6.22) ----------

LOUVAIN_SEED = 7   # repo convention: chunkgraph.py:732, graph3d.py:168


def node_dwpc(pathways: dict) -> dict:
    """P3 numerator: node_dwpc(o) = sum of pair['dwpc'] over every pathway pair
    whose BEST path contains chunk o. Chunks on no best path are absent (== 0.0).

    Spec: .spec/specs/graph-explorer/design.md 6.22 P3
    Task: playbook.md T39
    """
    out: dict = {}
    for p in (pathways.get("pairs") or []):
        dwpc = float(p["dwpc"])
        for o in (p.get("path") or []):
            out[o] = out.get(o, 0.0) + dwpc
    return out


def term_bm25_rank(members, sal) -> dict:
    """BM25 salience proxy for the P3 tie-break. gt.chunk_salient returns terms
    already ORDERED by BM25 descending but not the scores themselves, so
    salience is the summed normalised rank over the member chunks that carry
    the term: sum over o of (len(kept_o) - i) / len(kept_o) for the term at
    index i. Monotone in the per-chunk BM25 order, which is the only signal
    the gate exposes.

    Spec: .spec/specs/graph-explorer/design.md 6.22 P3
    Task: playbook.md T39
    """
    out: dict = {}
    for o in members:
        kept = sal.get(o, {}).get("kept") or []
        n = len(kept)
        if not n:
            continue
        for i, t in enumerate(kept):
            out[t] = out.get(t, 0.0) + (n - i) / n
    return out


def rank_group_terms(members, sal, ndw, k=12) -> list:
    """P3: term_dwpc(t) = sum of node_dwpc(o) over member chunks o whose
    salient vocab carries t. Sort key = (-term_dwpc, -bm25_rank, term): ties
    break by BM25 salience then lexicographic, and a zero-dwpc term therefore
    ranks below every positive term while keeping its BM25 order inside the
    zero class.

    Spec: .spec/specs/graph-explorer/design.md 6.22 P3
    Task: playbook.md T39
    """
    bm25 = term_bm25_rank(members, sal)
    term_dwpc: dict = {}
    term_chunks: dict = {}
    for o in members:
        kept = sal.get(o, {}).get("kept") or []
        dw = ndw.get(o, 0.0)
        for t in kept:
            term_dwpc[t] = term_dwpc.get(t, 0.0) + dw
            term_chunks.setdefault(t, []).append(o)
    rows = [{"term": t, "dwpc": term_dwpc[t], "bm25": bm25.get(t, 0.0),
             "chunks": term_chunks[t]} for t in term_dwpc]
    rows.sort(key=lambda r: (-r["dwpc"], -r["bm25"], r["term"]))
    return rows if k is None else rows[:k]


def subgraph_louvain(ords, edges, seed=LOUVAIN_SEED) -> dict:
    """P4 LEFT panel: louvain re-run on the walked subgraph only. `edges` is
    evidence.subgraph_edge_weights output (weight = strength). EPHEMERAL VIEW
    ONLY -- these ids are never persisted, never joined to community.cid,
    never carried across reruns (design 6.22 P4). Isolated chunks each get
    their own group.

    Spec: .spec/specs/graph-explorer/design.md 6.22 P4
    Task: playbook.md T39
    """
    ords = list(ords)
    if len(ords) < 2 or not edges:
        return {o: i for i, o in enumerate(sorted(ords))}

    import networkx as nx
    import community as community_louvain

    G = nx.Graph()
    G.add_nodes_from(ords)
    for (a, b), w in edges.items():
        G.add_edge(a, b, weight=w)
    part = community_louvain.best_partition(G, weight="weight", random_state=seed)

    groups: dict = {}
    for o, g in part.items():
        groups.setdefault(g, []).append(o)
    order = sorted(groups, key=lambda g: (-len(groups[g]), min(groups[g])))
    relabel = {g: i for i, g in enumerate(order)}
    return {o: relabel[g] for o, g in part.items()}


def group_entities(members, ents, floor=2, k=12) -> list:
    """P5: entities ranked by lift = (mention share in group) / (mention share
    in corpus), with a mention floor. group_share = sum(cnt over member
    chunks) / sum(cnt over member chunks, all entities); corpus_share =
    corpus[eid] / corpus_total. Entities with group_cnt < floor are dropped (a
    single mention gives an unbounded, meaningless lift). Sort
    (-lift, -cnt, name).

    Spec: .spec/specs/graph-explorer/design.md 6.22 P5
    Task: playbook.md T39
    """
    mentions = ents["mentions"]
    names = ents["names"]
    corpus = ents["corpus"]
    corpus_total = ents["corpus_total"]

    group_cnt: dict = {}
    for o in members:
        for eid, cnt in mentions.get(o, {}).items():
            group_cnt[eid] = group_cnt.get(eid, 0) + cnt
    group_total = sum(group_cnt.values())

    rows = []
    for eid, cnt in group_cnt.items():
        if cnt < floor:
            continue
        if group_total <= 0:
            continue
        corpus_cnt = corpus.get(eid, 0)
        if corpus_total <= 0 or corpus_cnt <= 0:
            continue
        group_share = cnt / group_total
        corpus_share = corpus_cnt / corpus_total
        lift = group_share / corpus_share
        rows.append({"entity_id": eid, "name": names.get(eid), "cnt": cnt,
                     "lift": lift, "group_share": group_share,
                     "corpus_share": corpus_share})
    rows.sort(key=lambda r: (-r["lift"], -r["cnt"], r["name"] or ""))
    return rows if k is None else rows[:k]


def group_relations(members, ents, rels, k=12) -> list:
    """P5: relation templates counted over pairs whose src AND dst entities
    are BOTH mentioned in this group's chunks. Counting is per
    (template, connector): `n` sums the corpus-level co-occurrence count,
    `pairs` counts distinct entity pairs that qualified inside the group.

    Spec: .spec/specs/graph-explorer/design.md 6.22 P5
    Task: playbook.md T39
    """
    mentions = ents["mentions"]
    present = {eid for o in members for eid in mentions.get(o, {})}

    agg: dict = {}
    for r in rels:
        if r["src"] not in present or r["dst"] not in present:
            continue
        key = (r["template"], r["connector"])
        a = agg.setdefault(key, {"n": 0, "pairs": set(), "top": []})
        a["n"] += r["n"]
        a["pairs"].add((r["src"], r["dst"]))
        a["top"].append((r["src_name"], r["dst_name"], r["llr"]))

    rows = []
    for (template, connector), a in agg.items():
        top = sorted(a["top"], key=lambda t: -t[2])[:3]
        rows.append({"template": template, "connector": connector, "n": a["n"],
                     "pairs": len(a["pairs"]), "top": top})
    rows.sort(key=lambda r: (-r["n"], -r["pairs"], r["template"], r["connector"]))
    return rows if k is None else rows[:k]


def group_classes(groups, ents, rels, sal, ndw, *, floor=2, k=12) -> list:
    """The one function walker_app (T40) calls per panel. `groups` is
    {ord: gid} (from subgraph_louvain for the relative panel, or
    Evidence.cid_of restricted to the walked chunks for the global panel --
    same shape, so ONE function serves both panels of P4). `sal` degrades to
    {} when ev.dendrite is None -- terms come back empty, entities/relations
    still populate, never a raise.

    Spec: .spec/specs/graph-explorer/design.md 6.22 P4/P5
    Task: playbook.md T39
    """
    by_group: dict = {}
    for o, gid in groups.items():
        by_group.setdefault(gid, []).append(o)

    rows = []
    for gid, members in by_group.items():
        members = sorted(members)
        rows.append({
            "gid": gid,
            "members": members,
            "size": len(members),
            "terms": rank_group_terms(members, sal, ndw, k=k),
            "entities": group_entities(members, ents, floor=floor, k=k),
            "relations": group_relations(members, ents, rels, k=k),
        })
    rows.sort(key=lambda r: (-r["size"], r["gid"]))
    return rows


def assess(conn, run, bundle, ev, ds, q, *, embed=None) -> tuple[dict, str | None]:
    """ONE model call (I13) over the walk, then the best-effort neo4j mirror.
    Returns (reason-result, mirror_error-or-None); the caller owns the spinner,
    the session_state write and how the error is shown.

    NEO4J_MIRROR=0 skips the mirror entirely (T17 behaviour, unchanged).
    """
    rr = interpret.reason(conn, run, bundle, ev.terms, ev.concept, embed=embed,
                          judge=True, pw=ev.pathways, digest=ev.digest)
    err = None
    if os.environ.get("NEO4J_MIRROR", "1") != "0":
        err = evidence.mirror_walk(bundle, ev, ds, prompt=q)
    return rr, err
