"""section_map.py -- the community map over whole SECTIONS: exact correlation edges in two representations, one fused graph, Leiden communities.

Spec: operator 2026-10-05 ("derive the graph based on sections"; "a model2vec version of that jina model applied to sections"; sparse vectors
with adjacent tokens; "I don't want to use hnsw anymore ... correlation matrix over both representations ... significant correlations as edges
for one graph ... louvain communities"); approved plan step B4 (C:\\Users\\user\\.claude\\plans\\jolly-soaring-moler.md). Task: playbook.md T154.

    SECTIONS   tools/section_corpus.py: the usable header blocks (80,642)
    DENSE      the saved jina-through-model2vec model pooled by tools/section_embed.py (.tmp/arxiv_sections_jina256.npy), centred on the corpus mean
    SPARSE     tools/section_sparse.py: BPE pieces + adjacent pairs, lambda 1, unit rows (.tmp/arxiv_sections_sparse_bpe.npz)
    EDGES      tools/section_graph.py: every pair exact, blockwise; significant = z >= k_sigma AND in a node's exact top-15; per space (cached)
    FUSE       the union of both spaces' significant edges and top-2 backbones, weight = mean z (clipped at 0) + a floor
    COMMUNITY  hnsw_communities.leiden over the fused matrix: resolution sweep, plateau pick, 3 x consensus, seed-ARI gate (arxiv_community_map's rules)
    LAYOUT     UMAP from the dense space's exact top-15 (no index)
    EXEMPLARS  arxiv_community_map.derive on sections (medoid + 1-2 picks by the size rule), the markdown and the PNG

Nothing here touches the chunk map: every output path is rebound to `.tmp/sections_*` before arxiv_community_map's writers run. Nothing is
written to Postgres (a later step, if wanted): the live `arxiv_sect` label and its build are untouched.

Run:  python -u tools\\section_map.py
"""
from __future__ import annotations

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import scipy.sparse as sp

import arxiv_community_map as m
import hnsw_communities as hc
import section_corpus as sc
import section_embed as se
import section_graph as sg
from community_exemplars import pick_resolution

T = ".tmp/"
DENSE_NPY, SPARSE_NPZ = T + "arxiv_sections_jina256.npy", T + "arxiv_sections_sparse_bpe.npz"
EDGES = {"dense": T + "dense_edges_xp.npz", "sparse": T + "sparse_edges_xp.npz"}          # cross-paper edges only; the paper-mixed ones are dense_edges.npz / sparse_edges.npz
OUT = {"OUT_EX": T + "sections_xpa_exemplars.json", "OUT_TOON": T + "sections_xpa_communities.toon", "OUT_MD": T + "sections_xpa_communities.md",
       "OUT_PNG": T + "sections_xpa_community_map.png", "SUMMARIES": T + "sections_xpa_cluster_summaries.json"}     # xpa = cross-paper edges, arm A (hierarchy, sqrt(m) bound)
STATE = T + "sections_xpa_map_state.npz"
TOPK, K_SIGMA = 15, 2.0
log = m.log


def self_first(knn_idx: np.ndarray, knn_sim: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Guarantee: (idx, dist) in the layout hnsw_communities.graph and UMAP's precomputed_knn expect: k columns, the node itself first at distance 0,
    then its k-1 strongest neighbours with dist = 1 - similarity."""
    N, k = knn_idx.shape
    idx = np.concatenate([np.arange(N)[:, None], knn_idx[:, :k - 1].astype(np.int64)], axis=1)
    dist = np.concatenate([np.zeros((N, 1), np.float32), (1.0 - knn_sim[:, :k - 1]).astype(np.float32)], axis=1)
    return idx, dist


def edges_for(name: str, X, path: str, topk: int = TOPK, group=None) -> tuple[sg.Cut, dict]:
    """Guarantee: (cut, edges) for representation `name`: the seeded null refitted (deterministic), the edges read from `path` when it holds this many
    rows, else computed exactly and written there. `group` (one integer per row, the paper) excludes same-paper pairs (section_graph.stream_edges)."""
    t0 = time.time()
    cut = sg.fit_null(X, rows=2000)
    log("%s null: lambda %.3f kurtosis %.3f threshold(z=%.0f) %.4f  %.0fs" % (name, cut.lam, cut.kurt, K_SIGMA, cut.threshold(K_SIGMA), time.time() - t0))
    if os.path.exists(path):
        z = np.load(path)
        if z["knn_idx"].shape[0] == X.shape[0]:
            log("%s edges cached: %d significant" % (name, len(z["i"])))
            return cut, {k: z[k] for k in z.files}
    e = sg.stream_edges(X, cut, k_sigma=K_SIGMA, topk=topk, block=512, log=lambda s: log("  %s %s" % (name, s)), group=group)
    np.savez(path, i=e["i"], j=e["j"], s=e["s"], knn_idx=e["knn_idx"], knn_sim=e["knn_sim"])
    log("%s edges: %s" % (name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in e["stats"].items() if k != "strongest_negative"}))
    return cut, e


def communities_from_graph(G: sp.csr_matrix, target_n: int = m.TARGET_N, fixed_res: float | None = None) -> dict:
    """Guarantee: {lab, chosen, cons_ari, res, n_comm, ari, plateau}: arxiv_community_map's rules on any weighted symmetric graph: a resolution sweep with seed
    stability, the plateau pick nearest the serving size, then the consensus of hc.SEEDS seeds, three times, whose mean pairwise ARI is the gate.
    With `fixed_res` the sweep and the pick are skipped and the consensus runs at that resolution (res, n_comm, ari, plateau are then None): the way to compare
    two graphs on one footing, since each graph's own plateau pick lands on a different resolution and with it a different community count (2026-10-06,
    arm C: 1.9 against 3.42)."""
    from sklearn.metrics import adjusted_rand_score as ari
    if fixed_res is None:
        res, n_c, a = m.sweep(G)
        i, plateau = pick_resolution(res, n_c, a, target_n)
        hc.RES = float(res[i])
        log("plateau at resolutions %s; serving size n=%d -> resolution %.3f (n~%.0f, seed-ARI %.3f)" % (np.round(res[plateau], 2).tolist(), target_n, hc.RES, n_c[i], a[i]))
    else:
        res = n_c = a = plateau = None
        hc.RES = float(fixed_res)
        log("resolution fixed at %.3f (no sweep, no plateau pick)" % hc.RES)
    labs = [hc.consensus(G, range(100 * r, 100 * r + hc.SEEDS)) for r in range(3)]
    cons = float(np.mean([ari(labs[p], labs[q]) for p in range(3) for q in range(p + 1, 3)]))
    log("consensus %d communities, seed-ARI %.3f (gate %.1f: %s)" % (labs[0].max() + 1, cons, m.GATE, "PASS" if cons >= m.GATE else "FAIL, labels must not be used for routing"))
    return {"lab": labs[0], "chosen": hc.RES, "cons_ari": cons, "res": res, "n_comm": n_c, "ari": a, "plateau": plateau}


def split_oversize(G: sp.csr_matrix, lab: np.ndarray, target_n: int = m.TARGET_N, bound: float | None = None,
                   max_depth: int = 3) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Guarantee: (labels, genre, report). A community holding more than `bound` members is re-partitioned INSIDE its own induced subgraph, with the
    same rules as the whole map: a resolution sweep, the plateau pick toward the serving size (members / (N / target_n)), then the consensus of
    hc.SEEDS seeds repeated three times whose mean pairwise seed-ARI must clear m.GATE. A split that fails the gate, or yields one part, is NOT applied
    (the community stays whole and the report says so). Each part larger than `bound` is split again, to at most `max_depth` levels. `genre` is the
    label BEFORE any split (the coarse level: a prompt-template community, a proofs community), `labels` the final one (the topic level). Parts get
    fresh ids after the existing ones; ids stay contiguous.
    `bound` defaults to sqrt(m), m = the graph's edge count (G.nnz / 2): the order of the modularity resolution limit (Fortunato and Barthelemy 2007),
    below which modularity cannot be relied on to separate communities. The constant is uncited, so it is a yardstick shared with the scorecard, not a
    derived fact; measured 2026-10-06 the repo's outlier rule (median + 2 MAD of log size) flags none of the five grab-bag communities of 1,525-2,324."""
    from sklearn.metrics import adjusted_rand_score as ari
    N = len(lab)
    mean = N / target_n
    bound = float(np.sqrt(G.nnz / 2)) if bound is None else float(bound)
    out, report, nxt = lab.copy(), [], int(lab.max()) + 1
    frontier = [(int(c), 0) for c in np.where(np.bincount(lab) > bound)[0]]
    while frontier:
        c, depth = frontier.pop(0)
        idx = np.where(out == c)[0]
        Gs = G[idx][:, idx].tocsr()
        want = max(2, int(np.ceil(len(idx) / mean)))
        res, n_c, a = m.sweep(Gs)
        i, _ = pick_resolution(res, n_c, a, want)
        hc.RES = float(res[i])
        labs = [hc.consensus(Gs, range(100 * r, 100 * r + hc.SEEDS)) for r in range(3)]
        cons = float(np.mean([ari(labs[p], labs[q]) for p in range(3) for q in range(p + 1, 3)]))
        parts = int(labs[0].max()) + 1
        ok = cons >= m.GATE and parts > 1
        report.append({"community": c, "depth": depth, "size": int(len(idx)), "wanted_parts": want, "resolution": hc.RES, "parts": parts,
                       "seed_ari": cons, "applied": bool(ok)})
        log("split community %d (%d, depth %d): resolution %.3f -> %d parts, seed-ARI %.3f (%s)" % (
            c, len(idx), depth, hc.RES, parts, cons, "applied" if ok else "NOT applied: gate or no split"))
        if not ok:
            continue
        ids = [c] + list(range(nxt, nxt + parts - 1))
        nxt += parts - 1
        for p in range(1, parts):
            out[idx[labs[0] == p]] = ids[p]
        if depth + 1 < max_depth:
            frontier += [(i_, depth + 1) for i_ in ids if (out == i_).sum() > bound]
    return out, lab.copy(), report


def run_names(genre_k: int = 0) -> tuple[dict, dict, str]:
    """Guarantee: (edge cache paths, output paths for arxiv_community_map's writers, state path) for a full run. genre_k = 0 is the arm A map (xpa, the names
    the module constants hold); genre_k > 0 is arm B at that k and gets its own `xpg<k>` names, so a run never overwrites another arm's files. The sparse
    edge cache is shared: the genre axes act on the dense view only."""
    if not genre_k:
        return EDGES, OUT, STATE
    tag = "xpg%d" % genre_k
    edges = {"dense": T + "dense_edges_%s.npz" % tag, "sparse": EDGES["sparse"]}
    out = {k: v.replace("sections_xpa_", "sections_%s_" % tag) for k, v in OUT.items()}
    return edges, out, T + "sections_%s_map_state.npz" % tag


def parse_run_args(argv: list[str]) -> int:
    """Guarantee: the genre k named by a `genre=K` argument (0 when none); anything else is an error."""
    bad = [a for a in argv if not a.startswith("genre=")]
    assert not bad and len(argv) <= 1, "usage: section_map.py [genre=K]; got %s" % argv
    return int(argv[0].split("=", 1)[1]) if argv else 0


def main(argv: list[str] | None = None) -> None:
    genre_k = parse_run_args(sys.argv[1:] if argv is None else argv)
    edges_paths, out_paths, state_path = run_names(genre_k)
    for k, v in out_paths.items():                             # the chunk map's writers now write the section map's files
        setattr(m, k, v)
    m.UNIT = "sections"
    recs = sc.load()
    N = len(recs)
    Xd, mu = se.center(np.load(DENSE_NPY))
    Z = sp.load_npz(SPARSE_NPZ).tocsr()
    assert Xd.shape[0] == N == Z.shape[0], (Xd.shape, N, Z.shape)
    log("%d sections | dense %s | sparse %s nnz/row %.0f | genre axes removed: %d" % (N, Xd.shape, Z.shape, Z.nnz / N, genre_k))
    paper = np.unique([r["doc_id"] for r in recs], return_inverse=True)[1]
    import section_genre as sgn
    Vg, _ = sgn.role_axes(Xd, paper, sgn.GENRE_K)                  # the genre FLAG always measures the ORIGINAL centred view, whatever arm builds the graph
    frac = sgn.genre_fraction(Xd, Vg)
    if genre_k:
        V, share = sgn.role_axes(Xd, paper, genre_k)
        Xd = sgn.remove_axes(Xd, V)
        log("removed %d within-paper axes carrying %.1f%% of the within-paper variance" % (genre_k, 100 * share.sum()))
    cd, ed = edges_for("dense", Xd, edges_paths["dense"], group=paper)
    cs, es = edges_for("sparse", Z, edges_paths["sparse"], group=paper)
    reps = {"dense": {"X": Xd, "cut": cd, "edges": ed}, "sparse": {"X": Z, "cut": cs, "edges": es}}
    f = sg.fuse(N, reps)
    both = ((f["prov"] & 1) > 0) & ((f["prov"] & 2) > 0)
    log("fused: %d edges | significant in dense %d, sparse %d, both %d | nodes with no significant edge in either space %d" % (
        len(f["i"]), int(((f["prov"] & 1) > 0).sum()), int(((f["prov"] & 2) > 0).sum()), int(both.sum()),
        int((np.bincount(np.concatenate([f["i"][f["prov"] > 0], f["j"][f["prov"] > 0]]), minlength=N) == 0).sum())))
    c = communities_from_graph(f["G"])
    c["lab"], c["genre"], split_report = split_oversize(f["G"], c["lab"])
    log("after splitting oversize communities: %d communities (%d at the genre level)" % (c["lab"].max() + 1, c["genre"].max() + 1))
    idx, dist = self_first(ed["knn_idx"], ed["knn_sim"])
    import umap
    t0 = time.time()
    XY = umap.UMAP(n_components=2, metric="cosine", precomputed_knn=(idx, dist), random_state=7).fit_transform(Xd)
    log("umap %.0fs" % (time.time() - t0))
    score, flag = sgn.genre_flags(c["lab"], frac)
    genre_set = {int(i) for i in np.where(flag)[0]}
    gsize = np.bincount(c["lab"])
    log("genre flag (mean genre fraction >= %.3f, communities of >= %d sections): %d communities, %d sections" % (
        sgn.GENRE_THRESHOLD, sgn.GENRE_MIN_SIZE, len(genre_set), int(gsize[sorted(genre_set)].sum()) if genre_set else 0))
    st = {"lab": c["lab"], "genre": c["genre"], "XY": XY, "idx": idx, "dist": dist, "chosen": c["chosen"], "cons_ari": c["cons_ari"],
          "genre_score": score, "genre_flag": flag}
    np.savez(state_path, **{k: v for k, v in st.items()})
    ex = m.derive(recs, Xd, Z, st, es["knn_idx"].astype(np.int64))
    n_sec, n_cut = m.write_sections_md(recs, recs, ex, {}, genre=genre_set)
    log("wrote %s (%d sections in full, %d cut)" % (out_paths["OUT_MD"], n_sec - n_cut, n_cut))
    m.render_cards(recs, st, ex, {}, genre=genre_set)
    sizes = np.bincount(c["lab"])
    log("DONE %d communities; sizes: max %d median %d, %d singletons" % (len(sizes), sizes.max(), int(np.median(sizes)), int((sizes == 1).sum())))


if __name__ == "__main__":
    main()
