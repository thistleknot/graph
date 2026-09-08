"""gt_terms.py -- vocabulary, salience, and the disk-cache layer for graph_tools.

Guards W8, W12 (tokenize half), W14, W17, W21 (expand_terms half) are stated
once in graph_tools.py; this module implements them.

Spec: .spec/specs/graph-explorer/design.md (module split, no behaviour change)
Task: playbook.md T37
"""
from __future__ import annotations

import math
import os
import re

import numpy as np
import psycopg

import config
from stoplist import _STOP                   # R18: one stoplist, no heavy imports


def _gt():
    """Sibling calls resolve through graph_tools at CALL time. Two reasons, both
    load-bearing: (1) the shim is the monkeypatch surface -- tests patch
    graph_tools.alias_map and graph_tools.CACHE_DIR and expect the call inside a
    sibling module to see it (test_graph_tools.py:1230, :652; test_sampler.py:860);
    (2) a call-time lookup makes the three modules acyclic and import-order-proof.
    After the first call this is a sys.modules dict hit."""
    import graph_tools
    return graph_tools


def tokenize(text: str) -> list[str]:
    """Mirrors chunkgraph._tok: lowercase alpha, >2 chars, stopwords dropped.
    Phrases are NOT merged -- the engine does not merge query phrases either,
    and diverging would silently change what is matched."""
    return [w for w in re.findall(r"[a-z]+", text.lower())
            if w not in _STOP and len(w) > 2]


def expand_terms(terms: list[str], aliases: dict) -> list[str]:
    """W21: alias expansion is ADDITIVE. A query token that IS an entity
    surface form contributes its whole alias set as extra OR-terms; every
    other token passes through untouched. Deterministic: sorted, deduped.

    Full weight, no discount: siblings enter the BM25 loop as ordinary terms
    with their own idf. BM25 already penalises a rarer alias through idf and
    saturates its tf, so a hand-picked expansion discount would be a second,
    unmeasured knob stacked on the one the scorer applies -- and this campaign
    has three wins for additive-never-displace and none for re-weighting.
    """
    out = set(terms)
    for t in terms:
        out.update(aliases.get(t, ()))
    return sorted(out)


_DF_CACHE: dict = {}
CACHE_DIR = config.CACHE_DIR


def _disk(key: str):
    """Runs are immutable (supersede-never-delete), so anything derived from a
    run_id can live on disk forever. Measured: the in-process index build is
    20 s on 500 documents; a pickle reload is well under a second."""
    import pickle
    path = os.path.join(_gt().CACHE_DIR, key + ".pkl")
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (OSError, EOFError, pickle.UnpicklingError):
        return None


def _disk_put(key: str, obj) -> None:
    import pickle, tempfile
    cache_dir = _gt().CACHE_DIR
    try:
        os.makedirs(cache_dir, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=cache_dir, suffix=".tmp")
        with os.fdopen(fd, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, os.path.join(cache_dir, key + ".pkl"))
    except OSError:
        pass                                         # cache is an optimisation, never a failure


def corpus_index(conn, run) -> dict:
    """Per-run inverted index, built ONCE and cached by run_id:
    {df: {term: df}, post: {term: {ord: tf}}, dl: {ord: n_tok}, avgdl, n}.
    One pass over the stored tf maps. Measured: the SQL BM25 over jsonb took
    7.3 s per query on 500 documents; this makes a query sub-millisecond."""
    key = str(run.run_id)
    if key not in _DF_CACHE:
        cached = _disk("index-" + key)
        if cached is not None:
            _DF_CACHE[key] = cached
            return cached
        post: dict = {}; dl: dict = {}
        with conn.cursor() as cur:
            cur.execute("""SELECT n.ord, (n.attrs->>'n_tok')::float AS dl,
                                  kv.key AS term, (kv.value)::float AS tf
                             FROM node n, jsonb_each_text(n.attrs -> 'tf') AS kv
                            WHERE n.run_id = %s""", (run.run_id,))
            for r in cur.fetchall():
                post.setdefault(r["term"], {})[r["ord"]] = r["tf"]
                dl[r["ord"]] = r["dl"] or 1.0
        n = len(dl) or 1
        _DF_CACHE[key] = {"df": {t: len(p) for t, p in post.items()}, "post": post,
                          "dl": dl, "avgdl": (sum(dl.values()) / n) if dl else 1.0, "n": n}
        _disk_put("index-" + key, _DF_CACHE[key])
    return _DF_CACHE[key]


_ALIAS_CACHE: dict = {}


def alias_map(conn, run) -> dict:
    """{name: (sibling, ...)} for every entity that has at least one alias.

    An alias set is exactly the rows sharing a canonical_id (entities.py E8).
    Siblings exclude the term itself and are sorted, so the map is a pure
    function of the run's rows.

    Degrades to {} where entity resolution has not run for this run -- a
    missing `entities` table, no rows, or canonical_id still NULL. That is a
    real state (T19 resolved planted fixtures only), and search must not
    depend on it.

    No disk cache. `_disk` is justified by runs being immutable; canonical_id
    is rewritten by every resolve_entities call, so this is not derived-from-
    an-immutable-run and must not outlive the process."""
    key = str(run.run_id)
    if key in _ALIAS_CACHE:
        return _ALIAS_CACHE[key]
    if len(_ALIAS_CACHE) > 32:
        _ALIAS_CACHE.clear()
    out: dict = {}
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT e.name AS name, s.name AS sibling
                  FROM entities e
                  JOIN entities s ON s.run_id = e.run_id
                                 AND s.canonical_id = e.canonical_id
                                 AND s.entity_id <> e.entity_id
                 WHERE e.run_id = %s AND e.canonical_id IS NOT NULL""",
                (run.run_id,))
            for r in cur.fetchall():
                out.setdefault(r["name"], set()).add(r["sibling"])
    except psycopg.Error:
        out = {}
    out = {name: tuple(sorted(sibs)) for name, sibs in out.items()}
    _ALIAS_CACHE[key] = out
    return out


def corpus_df(conn, run) -> tuple[dict, float, int]:
    """(df per term, avgdl, n_chunks) -- a view over corpus_index."""
    ix = corpus_index(conn, run)
    return ix["df"], ix["avgdl"], ix["n"]


def salient_gate(scores: dict) -> dict:
    """Which terms count as salient (W14). Log-normalise the BM25 scores, then
    keep every term at or above  min(median - 1.4826*MAD, mean - sd)  -- the
    more permissive of the robust and parametric centres, each one sigma down.
    Both branches are on the same sigma scale (1.4826 makes MAD normal-
    consistent). Measured: keeps ~84% on real chunks.

    Guarantee: {kept: [terms by score desc], threshold_decile: int in 1..10,
                n_in: int, n_kept: int}. Fewer than 4 terms: keep all."""
    if not scores:
        return {"kept": [], "threshold_decile": 0, "n_in": 0, "n_kept": 0}
    terms = sorted(scores, key=lambda t: (-scores[t], t))
    if len(terms) < 4:
        return {"kept": terms, "threshold_decile": 1, "n_in": len(terms), "n_kept": len(terms)}
    x = np.log1p(np.array([max(scores[t], 0.0) for t in terms], float))
    med = float(np.median(x)); mad = float(np.median(np.abs(x - med)))
    thr = min(med - 1.4826 * mad, float(x.mean() - x.std()))
    kept = [t for t, v in zip(terms, x) if v >= thr]
    below = int((x < thr).sum())
    decile = max(1, min(10, int(round(10 * below / len(x))) + 1))
    return {"kept": kept, "threshold_decile": decile, "n_in": len(terms), "n_kept": len(kept)}


def chunk_salient(conn, run, ords: list[int], k: int = 3,
                  K1: float = 1.5, B: float = 0.75) -> dict:
    """Per chunk: BM25 of its OWN terms against the corpus (chunk as document,
    idf over the run), gated by salient_gate (W14), top-k for the title.

    Guarantee: {ord: {top, kept, threshold_decile, n_in, n_kept}}."""
    if not ords:
        return {}
    df, avgdl, N = corpus_df(conn, run)
    out: dict = {}
    for o in ords:
        nd = _gt().node(conn, run, o)
        tf = (nd or {}).get("tf") or {}
        dl = float((nd or {}).get("n_tok") or 1)
        sc = {}
        for term, f in tf.items():
            d = df.get(term, 1); f = float(f)
            idf = math.log(1 + (N - d + 0.5) / (d + 0.5))
            sc[term] = idf * f * (K1 + 1) / (f + K1 * (1 - B + B * dl / avgdl))
        gate = salient_gate(sc)
        gate["top"] = gate["kept"][:k]
        out[o] = gate
    return out


def chunk_terms(conn, run, ords: list[int], k: int = 3) -> dict:
    """Top-k salient terms of each chunk -- what a medoid is titled with."""
    return {o: v["top"] for o, v in chunk_salient(conn, run, ords, k=k).items()}


# ---------------------------------------------------------------- dendrites
def dendrite_sort(M, names, alpha: float = 0.05, min_support: int = 0):
    """Correlation-chain decomposition -- the operator's 'dendrite sorting'
    (correlation sorting.md, 2026-09-02). Columns of M are the variables;
    rows are the observations (n for significance).

    Require: M is (n_obs, n_var) float array-like; names labels the columns.
    Guarantee: every kept variable lands in exactly ONE chain; chains are
      internally collinear threads (each hop is the tail's strongest
      significant unassigned partner), mutually decollinear groups (a new
      chain opens at the unassigned variable with the highest MEAN significant
      correlation -- the most connected remaining feature); a chain terminates
      WHEN no significant unassigned candidate remains. Deterministic:
      ties break by name order.
    Maintain: preprocessing per column is signed log1p -> Yeo-Johnson ->
      median/MAD z, so heavy-tailed, zero-heavy profiles (BM25 columns)
      correlate on rank-like scales. Columns with fewer than `min_support`
      nonzero observations are dropped before sorting (Pearson has nothing
      to grip on a near-empty profile).

    Returns {"chains": [[name,...],...], "r": (v,v) ndarray, "sig": bool
    ndarray, "names": kept names, "dropped": [...]} -- r/sig are aligned to
    the kept-name order.
    """
    from scipy.stats import t as _t, yeojohnson

    M = np.asarray(M, dtype=float)
    n_obs = M.shape[0]
    keep = [j for j in range(M.shape[1])
            if np.count_nonzero(M[:, j]) >= min_support and np.ptp(M[:, j]) > 0]
    dropped = [names[j] for j in range(M.shape[1]) if j not in set(keep)]
    names = [names[j] for j in keep]
    M = M[:, keep]
    v = len(names)
    if v == 0 or n_obs < 5:
        return {"chains": [], "r": np.zeros((0, 0)), "sig": np.zeros((0, 0), bool),
                "names": [], "dropped": dropped}

    P = np.empty_like(M)
    for j in range(v):
        x = np.sign(M[:, j]) * np.log1p(np.abs(M[:, j]))
        try:
            x = yeojohnson(x)[0]
        except Exception:
            pass                                     # degenerate: keep log scale
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        P[:, j] = (x - med) / (mad if mad else 1.0)

    R = np.atleast_2d(np.corrcoef(P, rowvar=False)) if v > 1 else np.zeros((1, 1))
    R = np.nan_to_num(R, nan=0.0)
    np.fill_diagonal(R, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        tt = np.abs(R) * np.sqrt((n_obs - 2) / np.maximum(1e-12, 1 - R ** 2))
    p = 2 * _t.sf(tt, df=n_obs - 2)
    sig = (p < alpha) & (R > 0)                       # chains ride positive links
    np.fill_diagonal(sig, False)

    # W23: significance is NECESSARY, not sufficient. At these sizes p<alpha
    # saturates -- measured on an 88-chunk walk, the p<.05 threshold is
    # r ~ 0.21 while the MEDIAN positive correlation is 0.59, so 57% of all
    # pairs qualified as a hop and the chain snaked through 87 of 88. A hop
    # must also clear the estimator-pair band on MAGNITUDE, derived from the
    # observed positive correlations (never hand-tuned):
    #     r >= max(mean + sdev, median + 1.4826*MAD)
    # the same ruler the NORMAL stage and A12's dilution detector use. On
    # that walk the band is 0.947 and 0.9% of pairs clear it -- threads, not
    # a snake. Degenerate case (fewer than 3 positive links) leaves `sig` as
    # significance alone rather than inventing a band from nothing.
    _pos = R[R > 0]
    if _pos.size >= 3:
        # The band is computed on the FISHER Z scale (arctanh), never on r.
        # Correlations are bounded at 1, so an additive band on a tight,
        # high-r distribution overshoots the maximum possible value: measured
        # on a live hurricane walk, mean+sdev gave band=1.041 -- no pair can
        # clear it, and every chunk fell out as a singleton (the opposite
        # failure to the snake). z = arctanh(r) is unbounded, so mean+-sdev
        # is meaningful there; tanh maps the threshold back.
        _z = np.arctanh(np.clip(_pos, -0.999999, 0.999999))
        _zmed = float(np.median(_z))
        _zmad = float(np.median(np.abs(_z - _zmed)))
        _band = float(np.tanh(min(float(_z.mean() + _z.std()),
                                  _zmed + 1.4826 * _zmad)))
        sig &= (R >= _band)
        np.fill_diagonal(sig, False)

    def mean_sig(j, pool):
        vals = [R[j, k] for k in pool if k != j and sig[j, k]]
        return float(np.mean(vals)) if vals else 0.0

    unassigned = set(range(v))
    chains = []
    while unassigned:
        master = max(sorted(unassigned), key=lambda j: (mean_sig(j, unassigned), -j))
        chain = [master]
        unassigned.remove(master)
        tail = master
        while True:
            cands = [k for k in sorted(unassigned) if sig[tail, k]]
            if not cands:
                break                                 # the chain terminates here
            tail = max(cands, key=lambda k: (R[tail, k], -k))
            chain.append(tail)
            unassigned.remove(tail)
        chains.append([names[j] for j in chain])
    return {"chains": chains, "r": R, "sig": sig, "names": names,
            "dropped": dropped}


def subgraph_embeddings(conn, run, ords: list[int]):
    """Stored dense embeddings for a set of ords, L2-normalized, as
    (ndarray, kept_ords). Ords without a stored embedding are omitted."""
    with conn.cursor() as cur:
        cur.execute("""SELECT ord, embedding::text AS e FROM node_embedding
                        WHERE run_id = %s AND ord = ANY(%s) ORDER BY ord""",
                    (run.run_id, list(ords)))
        rows = cur.fetchall()
    if not rows:
        return np.zeros((0, 0)), []
    E = np.array([np.fromstring(r["e"].strip("[]"), sep=",") for r in rows])
    E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    return E, [r["ord"] for r in rows]


# ------------------------------------------ second-order terms (W17)
def llr(k11, k12, k21, k22) -> float:
    """Dunning log-likelihood ratio (G2) on a 2x2 chunk-level co-occurrence
    table. Basis: .tmp/second_order_probe.py (measured probe, 2026-09-02);
    no governing REQ, W17 minted here."""
    def h(*ks):
        tot = sum(ks)
        return sum(k * math.log(k / tot) for k in ks if k > 0)
    return 2 * (h(k11, k12, k21, k22) - h(k11 + k12, k21 + k22)
                - h(k11 + k21, k12 + k22) + h(k11 + k12 + k21 + k22))


def second_order_terms(M, names, E, target, g2_gate: float = 10.83,
                       min_df: int = 5, skew_trip: float = 2.0,
                       min_skew_n: int = 8, top_k: int = 30) -> dict:
    """Which terms keep TARGET's company (W17). Pure, array-in/array-out --
    mirrors dendrite_sort's (M, names, ...) shape, no DB. Rows of M are
    chunks, columns are terms, nonzero = present; E is the matching chunk
    embedding matrix (or None/empty for a sparse-only run).

    Ladder: (1) Dunning-LLR gate on chunk-level co-occurrence admits only
    candidates significantly associated with the target and above min_df;
    (2) Schutze context-centroid cosine ranks the survivors by shared
    company; (3) a skew diagnostic over the survivors' cosine scores decides
    whether that ranking is trustworthy -- only when it trips does
    Mann-Whitney AUC re-rank the top_k. See W17.

    Basis: .tmp/second_order_probe.py (measured probe, 2026-09-02); no
    governing REQ, W17 minted here.

    Require: target in names, else LookupError.
    Guarantee: target never appears in its own ranked; sub-floor terms never
      enter the candidate pool and land in dropped (sorted); a sub-floor
      target returns ranked=[] with reason="target_df_below_min"; an empty
      candidate pool returns ranked=[] with reason="no_candidates"; every
      sort key is (-score, term) -- total and deterministic.
    """
    names = list(names)
    if target not in names:
        raise LookupError(f"target {target!r} not in vocabulary")
    ti = names.index(target)
    M = np.asarray(M)
    present = M != 0
    n_chunks = present.shape[0]
    df_arr = present.sum(axis=0)

    dropped = sorted(names[j] for j in range(len(names))
                     if j != ti and df_arr[j] < min_df)

    if df_arr[ti] < min_df:
        return {"target": target, "rung": "llr", "skew": 0.0, "ranked": [],
                "n_candidates": 0, "n_survivors": 0, "dropped": dropped,
                "reason": "target_df_below_min"}

    candidates = [j for j in range(len(names)) if j != ti and df_arr[j] >= min_df]
    n_candidates = len(candidates)
    if not candidates:
        return {"target": target, "rung": "llr", "skew": 0.0, "ranked": [],
                "n_candidates": 0, "n_survivors": 0, "dropped": dropped,
                "reason": "no_candidates"}

    tidx = np.nonzero(present[:, ti])[0]
    tset = set(tidx.tolist())
    N = n_chunks

    survivors = []                                    # (j, g2, member_idx)
    for j in candidates:
        idx = np.nonzero(present[:, j])[0]
        m = set(idx.tolist())
        k11 = len(tset & m)
        k12 = len(tset) - k11
        k21 = len(m) - k11
        k22 = N - k11 - k12 - k21
        g2 = llr(k11, k12, k21, k22)
        if g2 >= g2_gate:
            survivors.append((j, g2, idx))
    n_survivors = len(survivors)

    have_dense = E is not None and np.asarray(E).size > 0
    if n_survivors == 0 or not have_dense:
        ranked_rows = sorted(survivors, key=lambda x: (-x[1], names[x[0]]))[:top_k]
        ranked = [{"term": names[j], "g2": float(g2), "df": int(df_arr[j]),
                   "cos": None, "auc": None} for j, g2, _ in ranked_rows]
        return {"target": target, "rung": "llr", "skew": 0.0, "ranked": ranked,
                "n_candidates": n_candidates, "n_survivors": n_survivors,
                "dropped": dropped, "reason": None}

    E = np.asarray(E, dtype=float)
    E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)

    def centroid(idx):
        v = E[idx].mean(axis=0)
        n = np.linalg.norm(v)
        return v / (n if n > 1e-12 else 1.0)

    v_target = centroid(tidx)
    scored = [(j, g2, idx, float(v_target @ centroid(idx))) for j, g2, idx in survivors]
    ranked_centroid = sorted(scored, key=lambda x: (-x[3], names[x[0]]))

    cos_vals = np.array([x[3] for x in ranked_centroid])
    skew = 0.0
    if len(cos_vals) > min_skew_n and cos_vals.std() > 0:
        m3 = ((cos_vals - cos_vals.mean()) ** 3).mean() / cos_vals.std() ** 3
        n_ = len(cos_vals)
        skew = m3 * math.sqrt(n_ * (n_ - 1)) / max(n_ - 2, 1)
    trip = abs(skew) > skew_trip

    if not trip:
        rows = ranked_centroid[:top_k]
        ranked = [{"term": names[j], "g2": float(g2), "df": int(df_arr[j]),
                   "cos": cos, "auc": None} for j, g2, _, cos in rows]
        return {"target": target, "rung": "centroid", "skew": skew,
                "ranked": ranked, "n_candidates": n_candidates,
                "n_survivors": n_survivors, "dropped": dropped, "reason": None}

    top_rows = ranked_centroid[:top_k]
    is_t = np.zeros(N, dtype=bool)
    if tset:
        is_t[list(tset)] = True
    m_ = len(tset)
    reranked = []
    for j, g2, idx, cos in top_rows:
        s_all = E @ centroid(idx)
        r = s_all.argsort().argsort() + 1
        auc = float((r[is_t].sum() - m_ * (m_ + 1) / 2) / (m_ * (N - m_)))
        reranked.append((j, g2, cos, auc))
    reranked.sort(key=lambda x: (-x[3], names[x[0]]))
    ranked = [{"term": names[j], "g2": float(g2), "df": int(df_arr[j]),
               "cos": cos, "auc": auc} for j, g2, cos, auc in reranked]
    return {"target": target, "rung": "auc", "skew": skew, "ranked": ranked,
            "n_candidates": n_candidates, "n_survivors": n_survivors,
            "dropped": dropped, "reason": None}


def second_order(conn, run, target, ords: list[int] | None = None,
                 nomen_pool: list[str] | None = None, top_nomens: int = 400,
                 min_df: int = 5, max_df_frac: float = 0.5, k: int = 8) -> dict:
    """Thin DB-facing assembler for second_order_terms (W17): builds M,
    names, E from the corpus index and stored embeddings, then delegates.
    No ladder logic lives here.

    v0 nomen pool (when nomen_pool is None): terms with len > 2, not
    stoplisted, and df in [min_df, max_df_frac*n_chunks], ranked by df desc
    then term asc, truncated to top_nomens. This is a df-band + stoplist
    floor, NOT PPMI -- the upper max_df_frac band is the PPMI demotion's
    cheap proxy (a term carried by half the corpus co-occurs with
    everything, so its PMI against any target is near zero).
    # LATER: true PPMI demotion (entities v0 already computes ppmi in
    # entity_edges); swap this band for it once that table exists.
    Pass nomen_pool explicitly to bypass the floor entirely.

    Basis: .tmp/second_order_probe.py (measured probe, 2026-09-02); no
    governing REQ, W17 minted here.
    """
    ix = corpus_index(conn, run)
    if ords is None:
        ords = sorted(ix["dl"])
    E, kept = subgraph_embeddings(conn, run, ords)
    if kept:
        universe = kept
    else:
        universe = ords
        E = None

    if nomen_pool is None:
        n = ix["n"]
        cands = [t for t, d in ix["df"].items()
                 if len(t) > 2 and t not in _STOP and min_df <= d <= max_df_frac * n]
        cands.sort(key=lambda t: (-ix["df"][t], t))
        pool = cands[:top_nomens]
    else:
        pool = list(nomen_pool)

    names = list(pool)
    if target not in names:
        names = names + [target]
    post = ix["post"]
    M = np.zeros((len(universe), len(names)))
    for j, t in enumerate(names):
        p = post.get(t, {})
        for i, o in enumerate(universe):
            v = p.get(o)
            if v:
                M[i, j] = v

    result = second_order_terms(M, names, E, target, min_df=min_df)
    result["ranked"] = result["ranked"][:k]
    result["n_nomens"] = len(pool)
    result["universe"] = len(universe)
    return result
