"""arxiv_community_map.py -- the community map of the arxiv chunks, derived per sparsevec-lexsem-graph.

NO GOVERNING SPEC beyond that skill file (repo root, updated 2026-10-03) and operator instructions
2026-10-02/03 ("draw a community map ... top 3 chunks", "re-derive our setup and image with that
approach"). Reuses hnsw_communities.py unchanged (center, graph, leiden, consensus, n_per_community,
toon_table). Pure selection rules live in community_exemplars.py.

    STAGE      MECHANISM                                                                       GUARD
    LOAD       chunk records; is_reference and is_junk chunks EXCLUDED (each formed topics of its
               own); every derived cache is purged unless written for exactly these texts (A5)
    EMBED      all-MiniLM-L6-v2, fp16 GPU, 256-token window, mean-centered, L2; cached         R9, A1
    KNN        hnswlib cosine k=15, ONE insertion thread, built twice and asserted identical   A2
    SWEEP      10 log-spaced resolutions 0.1-20, 4 seeds: community count and seed-ARI         R20
    PICK       resolution inside the seed-ARI plateau whose n is nearest the serving size      R20
    CONSENSUS  16 seeds x 3 draws at that resolution; mean seed-ARI reported, gate 0.8         R11
    LAYOUT     UMAP 2D from the same kNN; labels never enter                                   R10
    LEXICAL    raw-word BM25 doc-doc view (idf baked, L2 rows, whales dropped), exact kNN      R7, A3
    EXEMPLARS  per community: band + lex-and-sem edge tiers, 1-3 picks at z = k/n, MMR         R23, R26, R28
    COLOURS    2D-neighbouring communities never share a hue; the 12 legend ones all differ    step 7
    PERSIST    dense vectors, communities, assignments, exemplars -> Postgres, on the live build (V7)
    OUTPUT     .tmp/arxiv_community_map.png, .tmp/arxiv_communities.toon, .tmp/arxiv_exemplars.json

A1  Embedding: all-MiniLM-L6-v2 beat static model2vec (128: 0.380, 256: 0.415, MiniLM 0.590 rec@50,
    tools/diag_dense_arms.py). nomic and EmbeddingGemma are NOT TESTED. The 256-token window sees only
    the opening of a long chunk.
A2  HNSW insertion with several threads is not deterministic: the same embeddings gave 148, 155 and 135
    communities in three runs. One thread makes the build reproducible; the run asserts it by building twice.
A3  The lexical view is raw words, not the skill's capped subword vocabulary: the sigma-rule subword arm
    has NOT been tested on this corpus (the earlier BPE-vs-raw comparison used a round-number vocabulary,
    which the skill closes). Terms in more than half the chunks are dropped from this view (bpe-bm25 step 7).
A5  A cache was valid when its row count matched, which is not an identity: removing junk while new
    papers arrive keeps the count and would serve the old embeddings, graph and lexical view as if
    freshly measured. The fingerprint is a hash of the texts in order (2026-10-03).
A7  The card text was laid out at a fixed figure height with a hard clip guard, so one longer summary
    (card 4 needed 4.38 in against 4.37) failed an otherwise finished rebuild. The figure now grows to
    its longest card (FIG_H is the floor, MAX_FIG_H the cap); beyond the cap it still raises.
A6  The final write used to crash the run when a viewer had the previous PNG open (2026-10-03, twice);
    save_figure falls back to a timestamped sibling and says so.
A4  TARGET_N is the serving size: the operator has been shown about 135 communities (~420 chunks each).

Run:  python -u tools\\arxiv_community_map.py      (delete .tmp/arxiv_map_state.npz to rebuild)
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import os
import pickle
import sys
import textwrap
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import scipy.sparse as sp

import hnsw_communities as hc
from community_exemplars import assign_colours, band, percentile_scaler, pick_resolution, select_exemplars
from domain_corpora import with_junk_flag

CACHE, EMB, STATE = ".tmp/arxiv_section_chunks.pkl", ".tmp/arxiv_emb_minilm.npy", ".tmp/arxiv_map_state.npz"
LEX, LEXKNN = ".tmp/arxiv_lex_view.npz", ".tmp/arxiv_lex_knn.npy"
KNN_SWEEP = ".tmp/arxiv_knn_sweep.npz"              # kNN graph + resolution sweep: slow, rule-free, cached
FP = ".tmp/arxiv_map_fingerprint.txt"               # which texts every cache above was written for
CACHES = (EMB, STATE, KNN_SWEEP, LEX, LEXKNN)
LABEL = "arxiv_sect"
OUT_PNG, OUT_TOON, OUT_EX = ".tmp/arxiv_community_map.png", ".tmp/arxiv_communities.toon", ".tmp/arxiv_exemplars.json"
SUMMARIES = ".tmp/arxiv_cluster_summaries.json"     # tools/summarize_clusters.py; LLM drafts, shown labelled as such
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
TARGET_N, N_RES, SWEEP_SEEDS, GATE, TOP_COMMS, FIG_H = 135, 10, 4, 0.8, 12, 21
MAX_FIG_H, CARD_MARGIN = 40, 0.2        # A7: the figure may grow to this height; a card keeps this much free below its text
PALETTE = ["#d55e00", "#0072b2", "#009e73", "#cc79a7", "#e69f00", "#56b4e9", "#7f3c8d", "#11a579",
           "#3969ac", "#f2b701", "#e73f74", "#80ba5a", "#a5aa99", "#e68310", "#008695", "#cf1c90"]


def log(msg: str) -> None:
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


# ------------------------------------------------------------------ inputs ----
def load() -> tuple[list[dict], list[int]]:
    """C9, C10. Guarantee: (retrievable records, their ords). References and junk are excluded: they
    would form topics of their own. An ord is the record's position among ALL chunks, the key
    lex_chunk_meta and every table keyed on it use."""
    allrecs = with_junk_flag(pickle.load(open(CACHE, "rb"))["records"])
    ords = [i for i, r in enumerate(allrecs) if not r["is_reference"] and not r["is_junk"]]
    log("chunks: %d retrievable of %d (%d references and %d junk excluded)" % (
        len(ords), len(allrecs), sum(r["is_reference"] for r in allrecs), sum(r["is_junk"] for r in allrecs)))
    return [allrecs[i] for i in ords], ords


def fingerprint(recs: list[dict]) -> str:
    """Guarantee: a hash of the chunk texts in order. Two runs share it only if they saw the same text."""
    h = hashlib.md5()
    for r in recs:
        h.update(hashlib.md5(r["text"].encode("utf-8")).digest())
    return h.hexdigest()


def purge_stale_caches(recs: list[dict], fp_path: str = FP, caches: tuple = CACHES) -> bool:
    """Guarantee: every derived cache is deleted unless it was written for exactly these texts in this
    order. A row count is not an identity -- junk removed while new papers arrive leaves the count
    alone -- and a cache that answers from the wrong texts looks like a fresh measurement. A cache
    from before fingerprints existed has no record and is purged. Returns True when it purged."""
    fp = fingerprint(recs)
    old = open(fp_path).read().strip() if os.path.exists(fp_path) else None
    if old == fp:
        return False
    for p in caches:
        if os.path.exists(p):
            os.remove(p)
    with open(fp_path, "w") as fh:
        fh.write(fp)
    return True


def encode_minilm(texts: list[str], step: int = 4000) -> np.ndarray:
    """A1, R9. Guarantee: (n, 384) float32 L2-normalised MiniLM embeddings, fp16 on the GPU with a
    256-token window -- the raw space `hc.center` takes its mean from."""
    import torch
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MINILM, device="cuda")
    m.max_seq_length = 256
    m.half()
    out, t0 = [], time.time()
    for i in range(0, len(texts), step):
        out.append(m.encode(texts[i:i + step], batch_size=64, normalize_embeddings=True,
                            show_progress_bar=False).astype(np.float32))
        log("embedded %d/%d  %.0fs" % (min(i + step, len(texts)), len(texts), time.time() - t0))
    del m
    torch.cuda.empty_cache()
    return np.concatenate(out)


def embed(recs: list[dict]) -> np.ndarray:
    if os.path.exists(EMB) and np.load(EMB).shape[0] == len(recs):
        log("embeddings cached")
        return np.load(EMB)
    E = encode_minilm([r["text"] for r in recs])
    np.save(EMB, E)
    return E


# ------------------------------------------------------------------- graph ----
def knn_det(E: np.ndarray, k: int = hc.K_NN):
    """A2. Guarantee: (idx, dist) like hc.knn, but built with ONE insertion thread so it is reproducible."""
    ix = hc.hnswlib.Index(space="cosine", dim=E.shape[1])
    ix.init_index(max_elements=len(E), M=16, ef_construction=200, random_seed=hc.HNSW_SEED)
    ix.add_items(E, np.arange(len(E)), num_threads=1)
    ix.set_ef(128)
    i, d = ix.knn_query(E, k=k)
    return i.astype(np.int64), d.astype(np.float32)


def sweep(G):
    """R20. Guarantee: (resolutions, mean community count, mean pairwise seed-ARI) over SWEEP_SEEDS seeds."""
    from sklearn.metrics import adjusted_rand_score as ari
    res = np.geomspace(0.1, 20, N_RES)
    n_c, a = np.zeros(N_RES), np.zeros(N_RES)
    for j, r in enumerate(res):
        t0 = time.time()
        labs = [hc.leiden(G, r, s) for s in range(SWEEP_SEEDS)]
        n_c[j] = np.mean([l.max() + 1 for l in labs])
        a[j] = np.mean([ari(labs[p], labs[q]) for p in range(SWEEP_SEEDS) for q in range(p + 1, SWEEP_SEEDS)])
        log("  res %6.3f  n %7.1f  seed-ARI %.3f  %.0fs" % (r, n_c[j], a[j], time.time() - t0))
    return res, n_c, a


def graph_and_sweep(E: np.ndarray):
    """Guarantee: (idx, dist, res, n_comm, seed_ari). The kNN graph and the sweep are the slow, rule-free
    part (about 6 min), so they are cached; changing the plateau rule or the serving size reuses them."""
    if os.path.exists(KNN_SWEEP):
        z = np.load(KNN_SWEEP)
        if z["idx"].shape[0] == len(E):
            log("kNN graph and resolution sweep cached")
            return z["idx"], z["dist"], z["res"], z["n_comm"], z["ari"]
    t0 = time.time()
    idx, dist = knn_det(E)
    idx2, _ = knn_det(E)
    assert (idx == idx2).all(), "single-thread HNSW build is not reproducible"
    G = hc.graph(idx, dist)
    log("knn built twice, identical; edges=%d  %.0fs" % (G.nnz // 2, time.time() - t0))
    log("resolution sweep (R20)")
    res, n_c, a = sweep(G)
    np.savez(KNN_SWEEP, idx=idx, dist=dist, res=res, n_comm=n_c, ari=a)
    return idx, dist, res, n_c, a


def build(E: np.ndarray) -> dict:
    idx, dist, res, n_c, a = graph_and_sweep(E)
    G = hc.graph(idx, dist)
    for r_, n_, a_ in zip(res, n_c, a):
        log("  sweep: res %6.3f  n %7.1f  seed-ARI %.3f" % (r_, n_, a_))
    i, plateau = pick_resolution(res, n_c, a, TARGET_N)
    chosen = float(res[i])
    log("plateau at resolutions %s; serving size n=%d -> resolution %.3f (n~%.0f, seed-ARI %.3f)"
        % (np.round(res[plateau], 2).tolist(), TARGET_N, chosen, n_c[i], a[i]))
    hc.RES = chosen
    from sklearn.metrics import adjusted_rand_score as ari
    t0 = time.time()
    labs = [hc.consensus(G, range(100 * r, 100 * r + hc.SEEDS)) for r in range(3)]
    cons = float(np.mean([ari(labs[p], labs[q]) for p in range(3) for q in range(p + 1, 3)]))
    log("consensus %d communities, seed-ARI %.3f (gate %.1f: %s)  %.0fs" % (
        labs[0].max() + 1, cons, GATE, "PASS" if cons >= GATE else "FAIL, labels must not be used for routing", time.time() - t0))
    import umap
    t0 = time.time()
    XY = umap.UMAP(n_components=2, metric="cosine", precomputed_knn=(idx, dist), random_state=7).fit_transform(E)
    log("umap %.0fs" % (time.time() - t0))
    return dict(lab=labs[0], XY=XY, idx=idx, dist=dist, res=res, n_comm=n_c, ari=a, plateau=plateau,
                chosen=chosen, cons_ari=cons)


# ------------------------------------------------------------ lexical view ----
def lexical_view(texts: list[str]):
    """R7, A3. Guarantee: CSR of L2-normalised idf-baked BM25 rows over raw words, whale terms dropped."""
    if os.path.exists(LEX):
        X = sp.load_npz(LEX).tocsr()
        if X.shape[0] == len(texts):
            log("lexical view cached %s" % (X.shape,))
            return X
    from salient_grams import bm25_matrix
    from stoplist import tokenize
    t0 = time.time()
    BM, _, terms, df = bm25_matrix([tokenize(t) for t in texts])
    BM = BM.tocsr().astype(np.float32)
    keep = np.where(df <= 0.5 * BM.shape[0])[0]
    BM = BM[:, keep]
    norm = np.sqrt(np.asarray(BM.multiply(BM).sum(1)).ravel())
    X = sp.diags(1 / np.maximum(norm, 1e-9)).dot(BM).tocsr().astype(np.float32)
    log("lexical view %s, %d whale terms dropped, %.0fs" % (X.shape, len(terms) - len(keep), time.time() - t0))
    sp.save_npz(LEX, X)
    return X


def lexical_knn(X, k: int = hc.K_NN, block: int = 1024, cache: str | None = LEXKNN) -> np.ndarray:
    """R7. Guarantee: exact top-k cosine neighbours (self excluded) per row of the lexical view."""
    if cache and os.path.exists(cache) and np.load(cache).shape[0] == X.shape[0]:
        return np.load(cache)
    N, XT, t0 = X.shape[0], X.T.tocsr(), time.time()
    idx = np.empty((N, k), np.int64)
    for s in range(0, N, block):
        S = (X[s:s + block] @ XT).toarray()
        S[np.arange(S.shape[0]), np.arange(s, s + S.shape[0])] = -1
        idx[s:s + S.shape[0]] = np.argpartition(-S, k, axis=1)[:, :k]
        if (s // block) % 10 == 0:
            log("  lexical kNN %d/%d  %.0fs" % (s, N, time.time() - t0))
    if cache:
        np.save(cache, idx)
    return idx


def null_cosines(E, X, n: int = 100_000, seed: int = 0):
    """R17, R28. Guarantee: (dense, lexical) cosines of n random distinct pairs: the null the percentiles are read on."""
    rng = np.random.default_rng(seed)
    i, j = rng.integers(0, len(E), n), rng.integers(0, len(E), n)
    ok = i != j
    i, j = i[ok], j[ok]
    dense = np.einsum("ij,ij->i", E[i], E[j])
    lex = np.concatenate([np.asarray(X[i[a:a + 20000]].multiply(X[j[a:a + 20000]]).sum(1)).ravel()
                          for a in range(0, len(i), 20000)])
    return dense, lex


# --------------------------------------------------------------- exemplars ----
def derive(recs, E, X, st, lidx) -> dict:
    """R23, R26, R28. Guarantee: {community: {size, n, exemplars[{role,row,z,key}], provenance, band}}."""
    lab, N = st["lab"], len(st["lab"])
    sizes = np.bincount(lab)
    n_show = hc.n_per_community(sizes)
    D = (hc.graph(st["idx"], st["dist"]) > 0).astype(np.int8)
    rows = np.repeat(np.arange(N), lidx.shape[1])
    L = sp.csr_matrix((np.ones(rows.size, np.int8), (rows, lidx.ravel())), shape=(N, N))
    A = D.multiply(L.maximum(L.T)).tocoo()                                    # edges present in BOTH views
    same = lab[A.row] == lab[A.col]
    has_edge = np.bincount(A.row[same], minlength=N) > 0
    log("lex-and-sem edges %d; members with one inside their community: %.1f%%" % (A.nnz // 2, 100 * has_edge.mean()))
    nd, nl = null_cosines(E, X)
    pct_d, pct_l = percentile_scaler(nd), percentile_scaler(nl)
    out = {}
    for c in range(len(sizes)):
        M = np.where(lab == c)[0]
        mu = E[M].mean(0)
        t = np.maximum(1 - E[M] @ mu / np.linalg.norm(mu), 0)

        def red(cands, shown, M=M):
            g, s = M[np.asarray(cands)], M[np.asarray(shown)]
            return (pct_d(E[g] @ E[s].T) * pct_l((X[g] @ X[s].T).toarray())).max(1)

        sel = select_exemplars(t, has_edge[M], int(n_show[c]), red)
        x, edge, info = band(t)
        papers = collections.Counter(recs[i]["doc_id"] for i in M)
        top_paper, top_n = papers.most_common(1)[0]
        out[c] = {"size": int(sizes[c]), "n": int(n_show[c]),
                  "exemplars": [{"role": r, "row": int(M[i]), "z": round(z, 3),
                                 "key": [recs[M[i]]["doc_id"], recs[M[i]]["section_idx"], recs[M[i]]["chunk_idx"]]}
                                for r, i, z in sel],
                  "n_papers": len(papers), "top_paper": top_paper, "top_share": round(top_n / len(M), 3),
                  "band": {**info, "share": round(float((x <= edge).mean()), 3),
                           "core_share": round(float(has_edge[M].mean()), 3)}}
    json.dump({str(c): v for c, v in out.items()}, open(OUT_EX, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return out


# ----------------------------------------------------------------- persist ----
def persist(ords: list[int], mean: np.ndarray, E: np.ndarray, st: dict, ex: dict, label: str = LABEL, conn=None) -> int:
    """V7. Guarantee: the label's live build holds this run's dense vectors, communities (unit centroids),
    consensus assignments with their UMAP position, and exemplars, replacing what an earlier run of this
    stage wrote on the same build. `mean` (the raw embedding mean E was centered on) goes into the build's
    params so a later chunk can be placed in the same space. Returns the build id. The build is opened by
    tools/ingest_arxiv_sparsevec.py, which fixes the statistics; none exists -> fail, never invent one."""
    import sparsevec_store as ss
    conn = conn or ss.connect()
    ss.ensure_build_schema(conn)
    live = ss.live_build(conn, label)
    assert live is not None, "no live build for %r: run tools/ingest_arxiv_sparsevec.py first" % label
    build, _, params = live
    lab, XY = st["lab"], st["XY"]
    assert len(ords) == len(E) == len(lab) == len(XY), "ords, vectors, labels and layout must be one row per chunk"
    ss.ensure_view_schema(conn, label, params["bm25"]["dim"], E.shape[1])
    ss.clear_derived(conn, build)
    ss.write_dense(conn, label, ords, E)
    ss.create_hnsw(conn, label, kind="dense")
    cents = []
    for c in range(int(lab.max()) + 1):
        mu = E[lab == c].mean(0)
        cents.append((c, int((lab == c).sum()), (mu / max(np.linalg.norm(mu), 1e-12)).tolist()))
    ss.write_communities(conn, build, cents)
    ss.write_assignments(conn, build, [(ords[i], int(lab[i]), "consensus", float(XY[i, 0]), float(XY[i, 1]))
                                       for i in range(len(lab))])
    ss.write_exemplars(conn, build, [(int(c), k, it["role"], it["z"], ords[it["row"]])
                                     for c, v in ex.items() for k, it in enumerate(v["exemplars"])])
    ss.update_build_params(conn, build, {
        "dense": {"model": MINILM, "dim": int(E.shape[1]), "max_seq_length": 256, "mean": mean.astype(float).tolist()},
        "communities": {"n": int(lab.max()) + 1, "resolution": float(st["chosen"]),
                        "consensus_seed_ari": float(st["cons_ari"]), "gate": GATE}})
    log("persisted build %d: %d vectors, %d communities, %d assignments, %d exemplars"
        % (build, len(E), len(cents), len(lab), sum(len(v["exemplars"]) for v in ex.values())))
    return build


# ------------------------------------------------------------------ output ----
def save_figure(fig, path: str, dpi: int = 80) -> str:
    """A6. Guarantee: the figure is on disk; returns the path written. Windows cannot truncate or replace a
    file another program holds memory-mapped (an image viewer showing it): open(w+b) raises Errno 22
    'Invalid argument' and os.replace raises Access is denied, both reproduced on a mapped copy. The
    stable name is tried first; when it is held, the figure goes to a timestamped sibling and the
    warning names both, so a finished rebuild is never lost at its last step to an open viewer."""
    try:
        fig.savefig(path, dpi=dpi)
        return path
    except OSError as exc:
        stem, ext = os.path.splitext(path)
        alt = "%s.%s%s" % (stem, time.strftime("%Y%m%d-%H%M%S"), ext)
        fig.savefig(alt, dpi=dpi)
        log("WARNING: %s is held open by another program (%s), probably an image viewer; wrote %s instead. "
            "Close the viewer and the next render updates the stable name." % (path, exc, alt))
        return alt


def mpl(s: str) -> str:
    """Guarantee: s safe for matplotlib text. '$' opens mathtext; '$$\\pi ^ { * } ...' crashed a render once."""
    return s.replace("$", r"\$")


def snippet(rec: dict, n: int = 150) -> str:
    body = rec["text"].split("\n", 1)[1] if "\n" in rec["text"] else ""
    s = " ".join(body.split())
    return s if len(s) <= n else s[:n - 3].rstrip() + "..."


def load_summaries(ex: dict) -> dict[int, dict]:
    """Guarantee: {community: draft} for summaries whose source chunks still ARE the community's exemplars.
    Anything else is stale (labels rebuilt, ids shifted): skipped and logged, never shown under the wrong cluster."""
    if not os.path.exists(SUMMARIES):
        return {}
    out, stale = {}, 0
    for d in json.load(open(SUMMARIES, encoding="utf-8")):
        c = d["community"]
        if d.get("status") == "draft" and c in ex and d["chunk_keys"] == [e["key"] for e in ex[c]["exemplars"]]:
            out[c] = d
        elif d.get("status") == "draft":
            stale += 1
    if stale:
        log("WARNING: %d summaries are stale (written for other exemplars) and are not shown" % stale)
    return out


def card_plan(k: int, c: int, e: dict, summ: dict | None, recs: list[dict]) -> tuple[list, float]:
    """A7. Guarantee: ([(text, y_inches_from_card_top, x0, text_kwargs)], total_inches). The whole summary is
    laid out, never cut for display; the caller sizes the figure to the longest card."""
    items, used = [], [0.08]

    def put(text, step, x0=0.02, **kw):
        items.append((text, used[0], x0, kw))
        used[0] += step

    put("%d  community %d  (%d chunks)" % (k + 1, c, e["size"]), 0.30, fontsize=11.5, fontweight="bold")     # colour: the community's
    if summ:
        put(summ["title"], 0.26, fontsize=10.5, fontweight="bold", color="#0b0b0b")
        for ln in textwrap.wrap(summ["summary"], 80):
            put(ln, 0.165, fontsize=8.8, color="#1a1a1a")
        used[0] += 0.06
    else:
        put("(no summary for this community)", 0.26, fontsize=9, color="#888888")
    put("%d papers; largest %s = %.0f%% | %d exemplar%s, z=k/n inside the band" % (
        e["n_papers"], e["top_paper"].replace("arxiv/", ""), 100 * e["top_share"], len(e["exemplars"]),
        "" if len(e["exemplars"]) == 1 else "s"), 0.2, fontsize=7.4, color="#666666")
    for item in e["exemplars"]:
        r = recs[item["row"]]
        put("%s  z=%.2f  %s  [%s]" % (item["role"], item["z"], textwrap.shorten(r["section_title"], 50, placeholder="..."),
                                      r["doc_id"].replace("arxiv/", "")), 0.19, fontsize=8, fontweight="bold", color="#0b0b0b")
        for ln in textwrap.wrap(snippet(r, 210), 74):
            put(ln, 0.16, x0=0.04, fontsize=7.6, color="#444444", family="monospace")
        used[0] += 0.05
    return items, used[0]


def render(recs, st, ex, summ):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lab, XY = st["lab"], st["XY"]
    sizes = np.bincount(lab)
    cen = np.array([np.median(XY[lab == c], axis=0) for c in range(len(sizes))])
    col = assign_colours(cen, sizes, len(PALETTE), k=6, fixed_top=TOP_COMMS)
    colour = lambda c: PALETTE[col[c]]
    top = list(np.argsort(-sizes, kind="stable")[:TOP_COMMS])
    cx0, cw, ch = 0.375, 0.200, 0.2175
    plans = {c: card_plan(k, c, ex[c], summ.get(c), recs) for k, c in enumerate(top)}
    need = max(total for _, total in plans.values())
    fig_h = max(FIG_H, math.ceil((need + CARD_MARGIN) / ch * 10) / 10)               # A7: the figure grows to its longest card
    if fig_h > MAX_FIG_H:
        worst = max(plans, key=lambda c: plans[c][1])
        raise ValueError("card %d text needs %.2f in, a %.1f in figure: it would be clipped above the %d in cap"
                         % (worst, plans[worst][1], fig_h, MAX_FIG_H))
    card_in = ch * fig_h
    fig = plt.figure(figsize=(30, fig_h), facecolor="#fcfcfb")
    ax = fig.add_axes([0.005, 0.02, 0.36, 0.93], facecolor="#fcfcfb")
    ax.scatter(XY[:, 0], XY[:, 1], s=2, c=[PALETTE[col[l]] for l in lab], linewidths=0, alpha=0.85)
    for c in range(len(sizes)):                                               # every community's medoid is a star
        r = ex[c]["exemplars"][0]["row"]
        ax.scatter(XY[r, 0], XY[r, 1], marker="*", s=40 if c not in top else 0, c=colour(c), edgecolors="#0b0b0b", linewidths=0.4)
    for k, c in enumerate(top):
        r = ex[c]["exemplars"][0]["row"]
        ax.text(XY[r, 0], XY[r, 1], str(k + 1), fontsize=14, fontweight="bold", color="#0b0b0b", ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec=colour(c), lw=2))
    ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
    ax.set_title("arXiv chunks: %d points, %d communities\n(resolution %.2f, consensus seed-ARI %.3f, gate %.1f %s)\nnumbered = the %d largest"
                 % (len(lab), len(sizes), st["chosen"], st["cons_ari"], GATE, "met" if st["cons_ari"] >= GATE else "NOT MET", TOP_COMMS),
                 fontsize=12.5, loc="left")
    for k, c in enumerate(top):
        cc, row = k % 3, k // 3
        x, y = cx0 + cc * (cw + 0.005), 0.975 - (row + 1) * (ch + 0.004) + 0.004
        card = fig.add_axes([x, y, cw, ch], facecolor="white")
        card.set_xticks([]); card.set_yticks([])
        [s.set_color(colour(c)) for s in card.spines.values()]
        card.set_xlim(0, 1); card.set_ylim(0, 1)
        items, total = plans[c]
        assert total <= card_in - CARD_MARGIN + 1e-9, "card %d would be clipped" % c
        for text, at, x0, kw in items:
            card.text(x0, 1 - at / card_in, mpl(text), va="top", **(kw if "color" in kw else {**kw, "color": colour(c)}))
        if summ.get(c):
            card.text(0.99, 0.012, "LLM draft | " + summ[c]["model"], fontsize=6.8, color="#888888", ha="right", va="bottom")
    log("wrote " + save_figure(fig, OUT_PNG))


def main():
    recs, ords = load()
    if purge_stale_caches(recs):
        log("caches were written for other texts: purged (%s)" % ", ".join(os.path.basename(p) for p in CACHES))
    raw = embed(recs)
    E = hc.center(raw, raw.mean(0))
    if os.path.exists(STATE) and np.load(STATE)["lab"].shape[0] == len(recs):
        z = np.load(STATE)
        st = {k: z[k] for k in z.files}
        st["chosen"], st["cons_ari"] = float(st["chosen"]), float(st["cons_ari"])
        log("state cached: %d communities, resolution %.3f, consensus seed-ARI %.3f; delete %s to rebuild"
            % (st["lab"].max() + 1, st["chosen"], st["cons_ari"], STATE))
    else:
        st = build(E)
        np.savez(STATE, **st)
    X = lexical_view([r["text"] for r in recs])
    lidx = lexical_knn(X)
    ex = derive(recs, E, X, st, lidx)
    persist(ords, raw.mean(0), E, st, ex)
    summ = load_summaries(ex)
    sizes = np.bincount(st["lab"])
    rows = [[c, ex[c]["size"], it["role"], it["z"], it["key"][0], it["key"][1], it["key"][2],
             recs[it["row"]]["section_title"], snippet(recs[it["row"]])]
            for c in np.argsort(-sizes, kind="stable") for it in ex[c]["exemplars"]]
    lines = hc.toon_table("communities", ["community", "size", "role", "z", "doc_id", "section_idx", "chunk_idx", "section_title", "snippet"], rows)
    if summ:
        srows = [[c, ex[c]["size"], "draft", summ[c]["model"], summ[c]["title"], summ[c]["summary"]]
                 for c in np.argsort(-sizes, kind="stable") if c in summ]
        lines += hc.toon_table("summaries", ["community", "size", "status", "model", "title", "summary"], srows)
    open(OUT_TOON, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    log("wrote %s (%d communities, %d summaries)" % (OUT_TOON, len(sizes), len(summ)))
    render(recs, st, ex, summ)


if __name__ == "__main__":
    main()
