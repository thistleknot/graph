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


# ---------- factbook digests (T43, design 6.22 P10-P12) ----------

_SEP = " . "


def _pack(items, width, *, sep=_SEP, prefix="", empty="-", max_items=None) -> str:
    """Greedy width packer, the single truncation rule for every rendered line
    (P11(f)). Keeps items, in order, only while the line so far PLUS a
    projected " + N more" tail still fits `width`; the rest collapse into
    that trailing count. `max_items` caps the list before packing even
    considers width (P11(d)'s per-panel cap) -- items past the cap fold into
    the same "+ N more" count as anything width drops. A single item that
    alone exceeds `width` is kept whole rather than cut mid-token (`clip`'s
    posture).

    Spec: .spec/specs/graph-explorer/design.md 6.22 P11
    Task: playbook.md T43
    """
    items = list(items)
    total = len(items)
    if max_items is not None and total > max_items:
        capped, overflow = items[:max_items], total - max_items
    else:
        capped, overflow = items, 0

    if not capped:
        return prefix + (f"+ {overflow} more" if overflow else empty)

    kept = []
    for item in capped:
        trial = kept + [item]
        remaining = overflow + (len(capped) - len(trial))
        line = prefix + sep.join(trial)
        probe = line + sep + f"+ {remaining} more" if remaining else line
        if len(probe) <= width or not kept:
            kept = trial
        else:
            break

    dropped = len(capped) - len(kept)
    remaining = overflow + dropped
    line = prefix + sep.join(kept)
    if remaining:
        line += sep + f"+ {remaining} more"
    return line


def group_digest(gc, ents, src_of=None, *, prefix="g", marker=None, width=110,
                  max_pairs=8) -> str:
    """Render ONE `group_classes` row as 4 TOON-style lines joined by "\\n":
    header, terms(dwpc), entities(mentions), relations. Pure/deterministic --
    same inputs produce a byte-identical string, so the UI can diff runs.

    Spec: .spec/specs/graph-explorer/design.md 6.22 P10/P11
    Task: playbook.md T43
    """
    members = gc.get("members") or []
    header = f"{prefix}{gc['gid']} . {gc['size']} chunks"
    if src_of is not None:
        mix = Counter(src_of.get(o) or "?" for o in members)
        mix_str = " ".join(f"{name}:{cnt}" for name, cnt in
                            sorted(mix.items(), key=lambda kv: (-kv[1], kv[0])))
        header += " . " + mix_str
    if marker is not None:
        header += " . " + marker

    terms = gc.get("terms") or []
    terms_line = _pack([f"{t['term']} {t['dwpc']:.1f}" for t in terms], width,
                        prefix="terms(dwpc): ")

    def _corpus_cnt(eid):
        c = ents.get("corpus", {}).get(eid)
        if c is not None:
            return c
        total = ents.get("corpus_total") or 0
        return round((row.get("group_share") or 0) * total)

    names = ents.get("names", {})
    ent_items = []
    for row in sorted(gc.get("entities") or [],
                       key=lambda e: (-e["cnt"], e.get("name") or e["entity_id"])):
        eid = row["entity_id"]
        name = row.get("name") or names.get(eid, eid)
        corpus_cnt = _corpus_cnt(eid)
        item = f"{name} {row['cnt']}"
        if corpus_cnt > row["cnt"]:
            item += f" x{row['lift']:.1f}"
        ent_items.append(item)
    ent_line = _pack(ent_items, width, prefix="entities(mentions): ")

    rel_items = []
    for r in gc.get("relations") or []:
        top = r.get("top") or []
        if top:
            a, b, _llr = top[0]
            rel_items.append(f"{a} -[{r['template']}]-> {b} corpus_n={r['n']}")
        else:
            rel_items.append(f"-[{r['template']}]-> corpus_n={r['n']}")
    rel_line = _pack(rel_items, width, prefix="relations: ", max_items=max_pairs)

    return "\n".join([header, terms_line, ent_line, rel_line])


def dedup_groups(rel_rows, glob_rows) -> list:
    """Merge the relative and global `group_classes` panels into ONE
    annotated list the UI iterates once, instead of rendering the same
    community twice (P11(c)). A relative group whose member set equals a
    global one is a straight rename; one that partially overlaps several
    globals gets a decomposition marker naming which globals it merges or
    splits; any global left untouched still renders on its own.

    Spec: .spec/specs/graph-explorer/design.md 6.22 P11(c)/(e)
    Task: playbook.md T43
    """
    gsets = {r["gid"]: set(r["members"]) for r in glob_rows}
    consumed = set()
    out = []

    for rel_row in rel_rows:
        S = set(rel_row["members"])
        equal_cid = None
        for cid, g in sorted(gsets.items()):
            if g == S:
                equal_cid = cid
                break
        if equal_cid is not None:
            consumed.add(equal_cid)
            out.append({"row": rel_row, "prefix": "g", "gid": rel_row["gid"],
                        "size": rel_row["size"], "kind": "merged",
                        "marker": f"= c{equal_cid} (global)",
                        "cids": [(equal_cid, len(S))]})
            continue

        inter = [(cid, len(S & g)) for cid, g in gsets.items() if S & g]
        inter.sort(key=lambda t: (-t[1], t[0]))
        partial = [cid for cid, n in inter if n < len(gsets[cid])]

        if partial:
            parts = " + ".join(f"c{cid}:{n}" for cid, n in inter)
            splits = ", ".join(f"c{c}" for c in partial)
            marker = f"= {parts} (splits {splits})"
        elif len(inter) > 1:
            marker = "= " + "+".join(f"c{cid}" for cid, _n in inter) + " (merges)"
        elif len(inter) == 1:
            marker = f"= c{inter[0][0]} (global)"
        else:
            marker = None

        out.append({"row": rel_row, "prefix": "g", "gid": rel_row["gid"],
                    "size": rel_row["size"], "kind": "local", "marker": marker,
                    "cids": inter})

    for glob_row in glob_rows:
        if glob_row["gid"] in consumed:
            continue
        out.append({"row": glob_row, "prefix": "c", "gid": glob_row["gid"],
                    "size": glob_row["size"], "kind": "global", "marker": None,
                    "cids": []})

    out.sort(key=lambda e: (-e["size"], e["prefix"], e["gid"]))
    return out


def chain_communities(chains, cid_of, *, width=110) -> list:
    """One line per dendrite chunk chain, parallel to `partition_rows`' rows:
    `"c6:31 c0:4 c12:1"`, counts descending then cid ascending. An ord absent
    from `cid_of` (or mapped to None) counts under the `"c?"` bucket, sorted
    last regardless of its count.

    Spec: .spec/specs/graph-explorer/design.md 6.22 P12
    Task: playbook.md T43
    """
    lines = []
    for chain in chains:
        counts = Counter()
        for o in chain:
            cid = cid_of.get(o)
            counts["c?" if cid is None else f"c{cid}"] += 1
        ordered = sorted(counts.items(),
                          key=lambda kv: (kv[0] == "c?", -kv[1], kv[0]))
        items = [f"{label}:{n}" for label, n in ordered]
        lines.append(_pack(items, width, sep=" "))
    return lines
