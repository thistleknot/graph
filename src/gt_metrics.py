"""gt_metrics.py -- adjacency numerics: centrality, pathways, partition metrics.

Guards W15, W16, W18, W19, W20 are stated once in graph_tools.py; this module
implements them.

Spec: .spec/specs/graph-explorer/design.md (module split, no behaviour change)
Task: playbook.md T37
"""
from __future__ import annotations

import math


def _gt():
    """Sibling calls resolve through graph_tools at CALL time. Two reasons, both
    load-bearing: (1) the shim is the monkeypatch surface -- tests patch
    graph_tools.alias_map and graph_tools.CACHE_DIR and expect the call inside a
    sibling module to see it (test_graph_tools.py:1230, :652; test_sampler.py:860);
    (2) a call-time lookup makes the three modules acyclic and import-order-proof.
    After the first call this is a sys.modules dict hit."""
    import graph_tools
    return graph_tools


BETWEENNESS_EXACT_MAX = 2000      # nodes; at or below this, betweenness is exact
BETWEENNESS_K         = 512       # pivot sample above the floor
BETWEENNESS_SEED      = 20260903  # fixed; the sample is a parameter, not a coin flip
PR_ALPHA, PR_TOL, PR_MAX_ITER  = 0.85, 1.0e-08, 200
PPR_ALPHA, PPR_TOL, PPR_MAX_ITER = 0.85, 1.0e-10, 200   # restart mass = 1 - PPR_ALPHA


def degrees(conn, run, ords: list[int]) -> dict:
    """Global degree of each ordinal in the run's live edge set (W15)."""
    if not ords:
        return {}
    with conn.cursor() as cur:
        cur.execute("""SELECT a AS ord, count(*) AS deg FROM edge_sym
                        WHERE run_id = %s AND valid_to IS NULL AND a = ANY(%s)
                        GROUP BY a""", (run.run_id, list(ords)))
        d = {r["ord"]: r["deg"] for r in cur.fetchall()}
    return {o: d.get(o, 0) for o in ords}


def pathways(conn, run, ords: list[int], anchors: list[int],
             max_len: int = 3, damp: float = 0.4, top_pairs: int = 12,
             ppr_alpha: float = PPR_ALPHA) -> dict:
    """W15 (design 6.11): critical connectedness between idea nodes.

    Induced subgraph = `ords` and the edges among them. Anchors are any
    ordinals in `ords` (medoids, top chunks -- the caller decides what an
    "idea" is). For every anchor pair: DWPC = sum over simple paths (<= max_len
    edges) of prod(edge strength) * prod(global deg(v)^-damp, every node on the
    path). Also returns the subgraph's shape: components, largest component
    share, density, and conductance of `ords` against the rest of the run.
    Deterministic; edge table only.
    """
    ords = list(dict.fromkeys(ords))
    anchors = [a for a in dict.fromkeys(anchors) if a in set(ords)]
    edges = _gt().subgraph_edges(conn, run, ords)
    adj: dict = {o: {} for o in ords}
    for e in edges:
        w = max(min(e["strength"], 1.0), 1e-9)
        adj[e["src"]][e["dst"]] = w
        adj[e["dst"]][e["src"]] = w
    deg = degrees(conn, run, ords)

    # ---- shape
    seen, comps = set(), []
    for o in ords:
        if o in seen:
            continue
        stack, comp = [o], set()
        while stack:
            u = stack.pop()
            if u in comp:
                continue
            comp.add(u)
            stack += [v for v in adj[u] if v not in comp]
        seen |= comp
        comps.append(len(comp))
    n, e_in = len(ords), len(edges)
    vol_s = sum(deg.values())
    with conn.cursor() as cur:
        cur.execute("""SELECT count(*) AS m FROM edge_sym
                        WHERE run_id = %s AND valid_to IS NULL""", (run.run_id,))
        m_total = cur.fetchone()["m"]          # directed rows = 2x undirected edges
    cut = max(vol_s - 2 * e_in, 0)
    vol_rest = max(m_total - vol_s, 1)
    comps.sort(reverse=True)
    shape = {"n": n, "edges": e_in, "components": len(comps),
             "wcc_sizes": comps,
             "largest_component_frac": (max(comps) / n) if n else 0.0,
             "density": (2 * e_in / (n * (n - 1))) if n > 1 else 0.0,
             "conductance": cut / min(max(vol_s, 1), vol_rest)}

    # ---- DWPC per anchor pair
    dmp = {o: (deg.get(o, 0) or 1) ** (-damp) for o in ords}
    pairs = []
    for i, a in enumerate(anchors):
        for b in anchors[i + 1:]:
            total, best, best_p, count = 0.0, 0.0, None, 0
            stack = [(a, [a], dmp[a])]
            while stack:
                u, path, sc = stack.pop()
                for v, w in adj[u].items():
                    if v in path:
                        continue
                    nsc = sc * w * dmp[v]
                    if v == b:
                        total += nsc
                        count += 1
                        if nsc > best:
                            best, best_p = nsc, path + [v]
                    elif len(path) <= max_len - 1:
                        stack.append((v, path + [v], nsc))
            if count:
                pairs.append({"a": a, "b": b, "dwpc": total, "n_paths": count,
                              "path": best_p})
    pairs.sort(key=lambda p: (-p["dwpc"], p["a"], p["b"]))

    # ---- W20: personalized PPR, additive -- DWPC still orders `pairs`
    node_ppr = ppr(adj, anchors, alpha=ppr_alpha)
    per_anchor = {a: ppr(adj, [a], alpha=ppr_alpha) for a in anchors}
    for pair in pairs:
        p_a, p_b = per_anchor.get(pair["a"], {}), per_anchor.get(pair["b"], {})
        pair["ppr"] = (p_a.get(pair["b"], 0.0) + p_b.get(pair["a"], 0.0)) / 2

    return {**shape, "pairs": pairs[:top_pairs], "ppr": node_ppr, "ppr_alpha": ppr_alpha}


_ADJ_CACHE: dict = {}


def full_adjacency(conn, run) -> dict:
    """W16: whole-run adjacency {a: {b: strength}}, cached per run (runs are
    immutable). ~2x edge count entries; fine to hold for graphs this size."""
    key = str(run.run_id)
    if key in _ADJ_CACHE:
        return _ADJ_CACHE[key]
    adj: dict = {}
    with conn.cursor() as cur:
        cur.execute("""SELECT a, b, strength FROM edge_sym
                        WHERE run_id = %s AND valid_to IS NULL""", (run.run_id,))
        for r in cur.fetchall():
            adj.setdefault(r["a"], {})[r["b"]] = max(min(r["strength"], 1.0), 1e-9)
    _ADJ_CACHE[key] = adj
    return adj


def best_path(conn, run, a: int, b: int, damp: float = 0.4):
    """W16 (design 6.13): the single strongest degree-damped path a..b over the
    WHOLE run -- Dijkstra minimising sum(-log strength) + damp*sum(log deg) over
    interior nodes, i.e. maximising the DWPC weight of one path. Returns
    (path list, weight in (0, 1]) or None when disconnected."""
    import heapq
    adj = full_adjacency(conn, run)
    if a not in adj or b not in adj:
        return None
    def node_cost(v):
        return damp * math.log(max(len(adj.get(v, {})), 1))
    dist = {a: 0.0}
    prev = {}
    heap = [(0.0, a)]
    seen = set()
    while heap:
        d, u = heapq.heappop(heap)
        if u in seen:
            continue
        seen.add(u)
        if u == b:
            break
        for v, w in adj[u].items():
            nd = d + -math.log(w) + (node_cost(v) if v != b else 0.0)
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    if b not in dist:
        return None
    path = [b]
    while path[-1] != a:
        path.append(prev[path[-1]])
    path.reverse()
    return path, math.exp(-dist[b])


# --------------------------------------- named graph metrics (W18/W19/W20)
def graph_metrics(adj: dict, k_sample: int | None = None,
                  seed: int = BETWEENNESS_SEED,
                  alpha: float = PR_ALPHA, tol: float = PR_TOL) -> dict:
    """W18. Centrality lane over {a: {b: strength}}. Pure -- no conn, unit-
    testable on planted data. Betweenness and clustering are UNWEIGHTED
    (networkx reads `weight` as a distance, and our strengths are
    similarities); PageRank is weighted, where higher strength genuinely
    means closer.

    Returns {"nodes": {ord: {"betweenness","pagerank","triangles",
    "clustering","degree"}}, "approx": bool,
    "params": {"k": k|None, "seed", "alpha", "tol"}, "n": int}.
    """
    import networkx as nx

    nodes = sorted(adj)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for a in nodes:
        for b in sorted(adj[a]):
            if a < b:
                G.add_edge(a, b, weight=adj[a][b])

    k = None if k_sample is None else min(k_sample, len(G))
    betweenness = nx.betweenness_centrality(G, k=k, seed=seed, weight=None,
                                            normalized=True)
    pagerank = nx.pagerank(G, alpha=alpha, tol=tol, max_iter=PR_MAX_ITER,
                           weight="weight")
    triangles = nx.triangles(G)
    clustering = nx.clustering(G, weight=None)
    approx = k is not None

    out_nodes = {o: {"betweenness": betweenness[o], "pagerank": pagerank[o],
                     "triangles": triangles[o], "clustering": clustering[o],
                     "degree": len(adj[o])} for o in nodes}
    return {"nodes": out_nodes, "approx": approx,
           "params": {"k": k, "seed": seed, "alpha": alpha, "tol": tol},
           "n": len(nodes)}


def _pct(sorted_xs: list, q: float) -> float:
    """Linear-interpolated percentile over an already-sorted list. 0.0 on
    empty input."""
    if not sorted_xs:
        return 0.0
    if len(sorted_xs) == 1:
        return float(sorted_xs[0])
    idx = q * (len(sorted_xs) - 1)
    lo, hi = int(math.floor(idx)), int(math.ceil(idx))
    if lo == hi:
        return float(sorted_xs[lo])
    frac = idx - lo
    return float(sorted_xs[lo] * (1 - frac) + sorted_xs[hi] * frac)


def partition_metrics(adj: dict, members: dict) -> dict:
    """W19. Density + conductance of the STORED partition -- members =
    {cid: [ord, ...]}; nothing is recomputed. Same algebra as pathways()'s
    shape block, deliberately, so the two numbers are comparable. Pure --
    no conn.

    Returns {"communities": [{cid,size,internal_edges,volume,cut,density,
    conductance}, ...] ordered by cid, "wcc": {"components","sizes",
    "largest_frac"}, "summary": {"density_median","density_p90",
    "conductance_median","conductance_p90","n_communities"}}.
    """
    e_total = sum(len(v) for v in adj.values()) / 2.0

    communities = []
    for cid in sorted(members):
        S = set(members[cid])
        n = len(S)
        volume = sum(len(adj.get(o, {})) for o in S)
        internal_edges = sum(1 for o in S for b in adj.get(o, {}) if b in S) // 2
        cut = max(volume - 2 * internal_edges, 0)
        vol_rest = max(2 * e_total - volume, 1)
        conductance = cut / min(max(volume, 1), vol_rest)
        density = (2 * internal_edges / (n * (n - 1))) if n > 1 else 0.0
        communities.append({"cid": cid, "size": n, "internal_edges": internal_edges,
                            "volume": volume, "cut": cut, "density": density,
                            "conductance": conductance})

    # ---- whole-run WCC (iterative DFS, no recursion -- mirrors pathways())
    seen, comps = set(), []
    for o in sorted(adj):
        if o in seen:
            continue
        stack, comp = [o], set()
        while stack:
            u = stack.pop()
            if u in comp:
                continue
            comp.add(u)
            stack += [v for v in adj[u] if v not in comp]
        seen |= comp
        comps.append(len(comp))
    comps.sort(reverse=True)
    n_total = len(adj)
    wcc = {"components": len(comps), "sizes": comps,
          "largest_frac": (max(comps) / n_total) if n_total else 0.0}

    densities = sorted(c["density"] for c in communities)
    conductances = sorted(c["conductance"] for c in communities)
    summary = {"density_median": _pct(densities, 0.5),
              "density_p90": _pct(densities, 0.9),
              "conductance_median": _pct(conductances, 0.5),
              "conductance_p90": _pct(conductances, 0.9),
              "n_communities": len(communities)}
    return {"communities": communities, "wcc": wcc, "summary": summary}


def ppr(adj: dict, seeds: list[int], alpha: float = PPR_ALPHA,
       tol: float = PPR_TOL, max_iter: int = PPR_MAX_ITER) -> dict:
    """W20. Personalized PageRank over {a:{b:w}} restarting on `seeds`
    (uniform personalization). Hand-rolled power iteration -- no nx import,
    keeping the additive pathways() column free of a new dependency on the
    hot path. Dangling nodes send their mass back to the restart vector.

    Returns {ord: score} summing to 1.0 over adj's keys; {} when seeds is
    empty, adj is empty, or none of seeds are in adj.
    """
    nodes = sorted(adj)
    seeds_in = [s for s in dict.fromkeys(seeds) if s in adj]
    if not nodes or not seeds_in:
        return {}
    n = len(nodes)
    idx = {o: i for i, o in enumerate(nodes)}
    p = [0.0] * n
    share0 = 1.0 / len(seeds_in)
    for s in seeds_in:
        p[idx[s]] = share0

    x = list(p)
    for _ in range(max_iter):
        new_x = [0.0] * n
        dangling_mass = 0.0
        for i, o in enumerate(nodes):
            neigh = adj[o]
            deg_w = sum(neigh.values())
            if deg_w <= 0:
                dangling_mass += x[i]
                continue
            share = alpha * x[i] / deg_w
            for b, w in neigh.items():
                j = idx.get(b)
                if j is not None:
                    new_x[j] += share * w
        restart_mass = (1 - alpha) + alpha * dangling_mass
        for i in range(n):
            new_x[i] += restart_mass * p[i]
        diff = sum(abs(new_x[i] - x[i]) for i in range(n))
        x = new_x
        if diff < tol:
            break
    return {nodes[i]: x[i] for i in range(n)}


_METRIC_CACHE: dict = {}
_COMM_METRIC_CACHE: dict = {}


def node_metrics(conn, run, ords: list[int] | None = None) -> dict:
    """W18. Whole-run centrality lane + per-provenance degree split, cached
    per run. `ords` filters the RETURNED view; it never changes what is
    computed."""
    gt = _gt()
    ck = str(run.run_id)
    disk_key = f"nodemetrics-{ck}-{BETWEENNESS_K}-{BETWEENNESS_SEED}"
    if ck not in _METRIC_CACHE:
        cached = gt._disk(disk_key)
        if cached is not None:
            _METRIC_CACHE[ck] = cached
    if ck not in _METRIC_CACHE:
        adj = full_adjacency(conn, run)
        k = None if len(adj) <= BETWEENNESS_EXACT_MAX else BETWEENNESS_K
        core = graph_metrics(adj, k_sample=k)
        with conn.cursor() as cur:
            cur.execute("""SELECT a AS ord, provenance, count(*) AS c FROM edge_sym
                            WHERE run_id = %s AND valid_to IS NULL
                            GROUP BY a, provenance""", (run.run_id,))
            prov_rows = cur.fetchall()
        prov: dict = {}
        for r in prov_rows:
            prov.setdefault(r["ord"], {})[r["provenance"]] = r["c"]
        for o, row in core["nodes"].items():
            p = prov.get(o, {})
            row["prov"] = p
            row["prov_degree_total"] = sum(p.values())
        _METRIC_CACHE[ck] = core
        gt._disk_put(disk_key, core)
    core = _METRIC_CACHE[ck]
    all_nodes = core["nodes"]
    if ords is None:
        view = dict(all_nodes)
    else:
        zero = {"betweenness": 0.0, "pagerank": 0.0, "triangles": 0,
               "clustering": 0.0, "degree": 0, "prov": {}}
        view = {o: all_nodes.get(o, dict(zero)) for o in ords}
    return {**core, "nodes": view}


def community_metrics(conn, run) -> dict:
    """W19. Per-stored-cid density + conductance, WCC sanity, run-wide
    median/p90."""
    gt = _gt()
    ck = str(run.run_id)
    disk_key = f"commmetrics-{ck}"
    if ck not in _COMM_METRIC_CACHE:
        cached = gt._disk(disk_key)
        if cached is not None:
            _COMM_METRIC_CACHE[ck] = cached
    if ck not in _COMM_METRIC_CACHE:
        with conn.cursor() as cur:
            cur.execute("""SELECT cid, members FROM community
                            WHERE run_id = %s ORDER BY cid""", (run.run_id,))
            rows = cur.fetchall()
        members = {r["cid"]: list(r["members"]) for r in rows}
        adj = full_adjacency(conn, run)
        result = partition_metrics(adj, members)
        _COMM_METRIC_CACHE[ck] = result
        gt._disk_put(disk_key, result)
    return _COMM_METRIC_CACHE[ck]
