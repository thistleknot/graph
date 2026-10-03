"""
quote_entity_clusters — dense kNN graph -> consensus-Leiden communities -> entities per community from a
draw-until sample -> text queries answered as a figure and as TOON text for an LLM. Single file.

STAGED PIPELINE
| # | Stage   | Step         | Mechanism                                                                              | Tag         | Guards |
|---|---------|--------------|----------------------------------------------------------------------------------------|-------------|--------|
| 1 | INGEST  | load         | quotes.jsonl, ftfy clean, keep texts within median +- 2 MAD of log2(length)             | [mandatory] | R1     |
| 2 | EMBED   | dense        | sentence-transformers (all-MiniLM-L6-v2), mean-centered, L2; cached in emb.npy           | [mandatory] | R2     |
| 3 | GRAPH   | knn          | hnswlib cosine k=15 -> symmetric weighted graph                                          | [mandatory] |        |
| 4 | COMM    | consensus    | 16 Leiden seeds (RB, res 3.0) -> co-association on kNN edges, keep >= 0.5 -> Leiden      | [mandatory] | R3     |
| 5 | LAYOUT  | umap2d       | UMAP 2D from the same kNN; labels never enter                                            | [mandatory] |        |
| 6 | ENRICH  | ner          | spaCy en_core_web_sm; one (text, class) per quote; numeric classes dropped               | [mandatory] | R4     |
| 7 | COUNT   | n_show       | Box-Cox of community sizes -> min-max -> round(1 + 2*scaled): quotes shown and entities listed per class | [opt] | R5     |
| 7b| LEX     | bm25         | BPE (8k) + BM25 doc vectors (idf-weighted, L2); lexical kNN graph; Gi = dense kNN x lexical kNN edges | [mandatory] | R9     |
| 8 | QUERY   | select       | query embedding -> top-N dense hits -> the TOP_K communities with most hits (>= MIN_HITS) | [opt]       | R6     |
| 8b| SELECT  | exemplars    | per selected community: medoid + picks at z = k/n on the centroid-distance band axis, MMR in equal-width windows | [opt] | R10    |
| 9 | SAMPLE  | draw_until   | uniform order inside the community; draw until M entity-bearing AND MIN_SAMPLE records, or exhausted; feeds 8b and entities | [opt]       | R7     |
|10 | REPORT  | png + toon   | map with selected communities; per community: query view (top hits) | community view (sample medoid + picks, z) | entities; TOON | [opt] | R8     |

GUARDS (EARS)
R1 Row count is not asserted; the filter and cleaning are the only text changes.
R2 Embeddings SHALL be centered with the corpus mean and re-normalised; queries SHALL use the same mean.
R3 Consensus SHALL be reported with seed-ARI over 3 independent draws; communities are for browsing and sampling, not ranking.
R4 An entity SHALL count once per quote; CARDINAL, ORDINAL, QUANTITY, PERCENT, MONEY SHALL be dropped.
R5 Entities per class SHALL be the top n_show by sample count, then by log-odds z vs the corpus; classes with none are omitted.
R6 Selection by hits SHALL report enrichment (hit share / size share) beside the hit count.
R7 Draws SHALL be uniform without replacement and SHALL NOT be weighted by hits; counts are x of the draws, never community rates.
R8 TOON SHALL quote a string only if empty, padded, true/false/null, number-like, or containing : , " \\ [ ] { } control
   characters, or starting with - or #; numbers (including numpy numbers) SHALL be bare.
R9 Redundancy SHALL be the product of percentile-scaled dense and lexical cosines (views on one scale before combining).
R10 A selected community SHALL be shown the way the map shows it: medoid first, then n-1 picks at equal steps k/n of the band position
    z = (t - t_medoid)/(edge - t_medoid) (t = Box-Cox distance to the centroid, edge = one-sided mean+1sd, median+1.4826 MAD only where
    skew remains), one MMR(lambda=.5) pick per equal-width window, pool = band AND lex∩sem edge (fallback band), never past the band.
R11 Exemplars and entities of a community SHALL come from ONE uniform sample of that community (>= MIN_SAMPLE records and >= ENT_M entity-bearing, or exhausted);
    only the lex∩sem edge marker reads the stored graph. The query's own view (top hits by score) SHALL be shown beside it, never merged into it.

CLOSED
- Fixed uniform-share quota for entity discovery: spent on records without entities; draw-until replaces it.
- Raw entity count alone as ranking: dominated by corpus-wide entities; enrichment z is shown beside it.
- Merging the query's hits into the community's exemplars: the hits are scored by the query, the exemplars are unbiased; two views, side by side (R11).
- Same-view cluster ids as a ranking filter or boost: neutral or worse; clusters here only select what to sample.

Preconditions: quotes.jsonl beside this file; pip: numpy scipy hnswlib igraph leidenalg umap-learn sentence-transformers
spacy (+ en_core_web_sm) ftfy matplotlib. Offline: set DENSE to a local model directory and HF_HUB_OFFLINE=1.
Run: python quote_entity_clusters.py            (writes query_entities.png, query_results.toon); tables per query: hits, hit_entities, clusters, community_view, community_entities
"""
import os, re, sys, json, math, textwrap, collections, numpy as np, scipy.sparse as sp
import ftfy, hnswlib, igraph, leidenalg, umap, spacy, tempfile
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
from scipy.stats import boxcox, skew
from sklearn.metrics import adjusted_rand_score as ari
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

DENSE = os.environ.get("DENSE", "sentence-transformers/all-MiniLM-L6-v2")
K1, B_BM25, VOCAB, PREFIX = 1.2, 0.75, 8000, "##"; LAM_MMR = 0.5; K_NN = 15; RES = 3.0; SEEDS = 16; THR = 0.5; HNSW_SEED = 7
TOP_N = 50; TOP_K = 3; MIN_HITS = 3; ENT_M = 10; MIN_SAMPLE = 30; PRIOR = 10.0; MAX_CLASSES = 4
DROP = {"CARDINAL", "ORDINAL", "QUANTITY", "PERCENT", "MONEY"}
ABBR = {"WORK_OF_ART": "ART", "PRODUCT": "PROD", "LANGUAGE": "LANG"}
QUERIES = ["Dumbledore on choosing between what is right and what is easy",
           "how time passes, every day, yesterday and tomorrow",
           "reading books and the love of stories"]
HI = ["#d55e00", "#0072b2", "#009e73"]; GREY = "#d6d5cf"; INK = "#0b0b0b"; INK2 = "#52514e"


# ---------- INGEST / EMBED / GRAPH / COMMUNITIES ----------
def load_texts(path="quotes.jsonl"):
    """Guarantee: list of cleaned texts inside median +- 2 MAD of log2 length (R1)."""
    t = [ftfy.fix_text(json.loads(l)["quote"]).strip().strip("\"' ").strip() for l in open(path)]
    x = np.log2([max(len(s), 1) for s in t]); med = np.median(x); mad = np.median(np.abs(x - med))
    return [s for s, v in zip(t, x) if med - 2 * mad <= v <= med + 2 * mad]


def embed_raw(texts, cache="emb.npy"):
    """Guarantee: raw L2-normalised embeddings, cached by corpus size."""
    if os.path.exists(cache) and np.load(cache).shape[0] == len(texts): return np.load(cache)
    from sentence_transformers import SentenceTransformer
    E = SentenceTransformer(DENSE).encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    np.save(cache, E); return E


def center(X, mu):
    """R2. Guarantee: rows centered with mu and L2-normalised."""
    X = X - mu; return (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)


def knn(E, k=K_NN):
    """Guarantee: (idx, dist) from an HNSW cosine index, k neighbours including self."""
    ix = hnswlib.Index(space="cosine", dim=E.shape[1]); ix.init_index(max_elements=len(E), M=16, ef_construction=200, random_seed=HNSW_SEED)
    ix.add_items(E, np.arange(len(E))); ix.set_ef(128); i, d = ix.knn_query(E, k=k); return i.astype(np.int64), d.astype(np.float32)


def graph(idx, dist):
    N, k = idx.shape; G = sp.csr_matrix((1 - dist[:, 1:].ravel(), (np.repeat(np.arange(N), k - 1), idx[:, 1:].ravel())), shape=(N, N)); return G.maximum(G.T)


def leiden(G, res, seed):
    T = sp.triu(G.tocsr(), 1).tocoo(); g = igraph.Graph(n=G.shape[0], edges=list(zip(T.row.tolist(), T.col.tolist()))); g.es["weight"] = T.data.tolist()
    return np.array(leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, weights="weight", resolution_parameter=float(res), seed=seed).membership)


def consensus(G, seeds):
    """R3. Guarantee: labels from Leiden on the co-association of kNN edges agreeing in >= THR of the seeds."""
    C = G.tocoo(); acc = np.zeros(C.nnz)
    for s in seeds: l = leiden(G, RES, s); acc += (l[C.row] == l[C.col])
    w = acc / len(seeds); k = w >= THR
    return leiden(sp.csr_matrix((w[k], (C.row[k], C.col[k])), shape=G.shape), RES, 0)


def n_per_community(sizes):
    """R5. Guarantee: int in [1, 3] per community from Box-Cox of size, min-max scaled."""
    bc, _ = boxcox(sizes.astype(float)); q = (bc - bc.min()) / (bc.max() - bc.min()); return np.minimum(np.rint(1 + 2 * q).astype(int), sizes)


# ---------- LEXICAL VIEW ----------
_NORM = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])


def train_bpe(texts, vocab=VOCAB):
    """Guarantee: BPE tokenizer (NFKC + lowercase + whitespace) where every corpus character exists bare and with the ## prefix (no [UNK])."""
    tok = Tokenizer(models.BPE(unk_token="[UNK]")); tok.normalizer = _NORM; tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=vocab, special_tokens=["[UNK]"], continuing_subword_prefix=PREFIX, show_progress=False))
    d = tempfile.mkdtemp(); vf, mf = tok.model.save(d); v = json.load(open(vf, encoding="utf8"))
    for ch in sorted({c for t in texts for c in _NORM.normalize_str(t) if not c.isspace()}):
        for f in (ch, PREFIX + ch):
            if f not in v: v[f] = len(v)
    json.dump(v, open(vf, "w", encoding="utf8"), ensure_ascii=False)
    tok.model = models.BPE.from_file(vf, mf, unk_token="[UNK]", continuing_subword_prefix=PREFIX); return tok


def lex_vectors(texts):
    """R9. Guarantee: dense float32 (N x V) idf-weighted BM25 doc vectors, rows L2-normalised (doc side saturated tf, idf applied once)."""
    tok = train_bpe(texts); ids = [e.ids for e in tok.encode_batch(texts)]; N, V = len(ids), tok.get_vocab_size()
    dl = np.array([len(i) for i in ids], float); r, c, v = [], [], []
    for d, toks in enumerate(ids):
        for t, tf in collections.Counter(toks).items(): r.append(d); c.append(t); v.append(tf)
    TF = sp.csr_matrix((v, (r, c)), shape=(N, V)); df = np.asarray((TF > 0).sum(0)).ravel(); idf = np.log(1 + (N - df + 0.5) / (df + 0.5))
    T = TF.tocoo(); sat = T.data * (K1 + 1) / (T.data + K1 * (1 - B_BM25 + B_BM25 * dl[T.row] / dl.mean()))
    X = sp.csr_matrix((sat, (T.row, T.col)), shape=(N, V)).multiply(idf).tocsr()
    X = sp.diags(1 / np.sqrt(X.multiply(X).sum(1).A.ravel())) @ X
    return X.toarray().astype(np.float32)


def pair_cos(X, a, b, chunk=4000):
    """Guarantee: row-wise dot of X[a], X[b] in chunks (bounded memory)."""
    return np.concatenate([(X[a[i:i + chunk]] * X[b[i:i + chunk]]).sum(1) for i in range(0, len(a), chunk)])


def pct(x, ref): return np.searchsorted(ref, x) / len(ref)


# ---------- EXEMPLARS (same rule as the cluster map) ----------
def centroid_axis(cc):
    """R10. Require: cc cosine to the community centroid. Guarantee: (t, edge, band): t = 1 - cc, Box-Cox where |skew| > 0.5;
    edge = mean + 1sd of t (median + 1.4826 MAD where |skew(t)| stays > 0.5); band = t <= edge (one-sided)."""
    d = np.clip(1 - cc, 1e-9, None)
    if len(d) < 4: return d, float(d.max()), np.ones(len(d), bool)
    t = boxcox(d)[0] if abs(skew(d)) > 0.5 else d
    edge = (np.median(t) + 1.4826 * np.median(np.abs(t - np.median(t)))) if abs(skew(t)) > 0.5 else t.mean() + t.std()
    return t, float(edge), t <= edge


def band_position(t, edge, med):
    """R10. Guarantee: z = (t - t_med)/(edge - t_med) floored at 0; zeros where the edge does not exceed the medoid."""
    span = edge - t[med]
    return np.zeros(len(t)) if span <= 1e-12 else np.clip((t - t[med]) / span, 0, None)


def pick_exemplars(z, red, med, pool, n):
    """R10. Guarantee: <= n local indices, medoid first, then one per target k/n (k=1..n-1) from the window |z - k/n| <= 1/(2n) by
    MMR(LAM_MMR) with relevance = closeness to the target and redundancy red; an empty window falls back to the nearest-z pool member."""
    picks = [med]; half = 1 / (2 * n)
    for k in range(1, n):
        tgt = k / n; cand = [i for i in pool if i not in picks]
        if not cand: break
        win = [i for i in cand if abs(z[i] - tgt) <= half] or [min(cand, key=lambda i: abs(z[i] - tgt))]
        picks.append(int(max(win, key=lambda i: LAM_MMR * max(0.0, 1 - abs(z[i] - tgt) / half) - (1 - LAM_MMR) * red[i, picks].max())))
    return picks


def exemplars(smp, m_all, E, L, Gi, refs, n):
    """R9, R10, R11. Require: smp global indices of a uniform sample of ONE community, m_all all its members (used only for the
    lex∩sem edge marker). Guarantee: (global indices medoid first, z per pick, pool kind); everything else is computed from the sample."""
    S = E[smp] @ E[smp].T; li = int(np.argmax(S.mean(1)))
    cen = E[smp].mean(0); cen /= np.linalg.norm(cen); t, edge, band = centroid_axis(E[smp] @ cen); z = band_position(t, edge, li)
    lexm = np.asarray((Gi[smp][:, m_all] > 0).sum(1)).ravel() > 0; pool, kind = np.where(band & lexm)[0], "core"
    if len(pool) < n: pool, kind = np.where(band)[0], "band"
    red = pct(S, refs[0]) * pct(L[smp] @ L[smp].T, refs[1])
    loc = pick_exemplars(z, red, li, list(pool), int(min(n, max(len(pool), 1))))
    return [int(smp[i]) for i in loc], [float(z[i]) for i in loc], kind


# ---------- ENTITIES / SAMPLER ----------
def entities(texts):
    """R4. Guarantee: list (per text) of sets of (lowercased text, class)."""
    nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
    return [{(x.text.strip().lower(), x.label_) for x in d.ents if x.label_ not in DROP and x.text.strip()} for d in nlp.pipe(texts, batch_size=64)]


def doc_freq(ents, idx):
    c = collections.Counter()
    for i in idx: c.update(ents[i])
    return c


def enrichment(df_c, n_c, df_all, N):
    """Guarantee: {entity: z}, log-odds in-community vs rest with a corpus-rate prior (Monroe et al.)."""
    z = {}
    for e, y in df_c.items():
        yr = df_all[e] - y; nr = N - n_c; a = PRIOR * df_all[e] / N
        lc = math.log((y + a) / (n_c - y + PRIOR - a)); lr = math.log((yr + a) / (nr - yr + PRIOR - a))
        z[e] = (lc - lr) / math.sqrt(1 / (y + a) + 1 / (n_c - y + PRIOR - a) + 1 / (yr + a) + 1 / (nr - yr + PRIOR - a))
    return z


def draw_until(idx, ents, m, rng, min_n=0):
    """R7. Guarantee: (sample indices, entity-bearing count seen) drawn uniformly without replacement until m entity-bearing records
    are seen AND at least min_n records are drawn, or the community is exhausted."""
    order = rng.permutation(idx); seen = 0
    for k, i in enumerate(order, 1):
        seen += bool(ents[i])
        if seen >= m and k >= min_n: return order[:k], seen
    return order, seen


def community_view(c, lab, E, L, Gi, refs, n_show, ents, df_all, rng):
    """R7, R10, R11. Guarantee: the unbiased view of community c from ONE uniform sample: sample indices, exemplar picks (medoid
    first) with their band positions z, pool kind, and entity rows (class, entity, count) top n per class by count then enrichment z."""
    m = np.where(lab == c)[0]; smp, _ = draw_until(m, ents, ENT_M, rng, MIN_SAMPLE); n = int(n_show[c])
    picks, zs, kind = exemplars(smp, m, E, L, Gi, refs, n)
    ds = doc_freq(ents, smp); zz = enrichment(doc_freq(ents, m), len(m), df_all, len(lab)); by = collections.defaultdict(list)
    for e, v in ds.items(): by[e[1]].append((v, zz[e], e[0]))
    ent = []
    for cls, Lc in sorted(by.items(), key=lambda kv: -sum(v for v, _, _ in kv[1]))[:MAX_CLASSES]:
        ent += [(cls, t, v) for v, _, t in sorted(Lc, reverse=True)[:n]]
    return dict(m=m, smp=smp, picks=picks, z=zs, kind=kind, ent=ent)


# ---------- TOON ----------
def toon_str(s):
    """R8. Guarantee: s as a TOON scalar, quoted and escaped only when the spec requires it."""
    s = str(s)
    need = (s == "" or s != s.strip() or s in ("true", "false", "null") or re.fullmatch(r"-?\d+(\.\d+)?([eE][+-]?\d+)?", s) is not None
            or re.search(r'[:",\\\[\]{}\x00-\x1f]', s) is not None or s.startswith(("-", "#")))
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t") + '"' if need else s


def toon_val(v):
    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool): return format(float(v), "g") if isinstance(v, (float, np.floating)) else str(v)  # canonical: 0.0 -> 0
    return toon_str(v)


def toon_table(name, fields, rows, pad=0):
    sp_ = " " * pad
    return [f"{sp_}{name}[{len(rows)}]{{{','.join(fields)}}}:"] + [sp_ + "  " + ",".join(toon_val(v) for v in r) for r in rows]


# ---------- QUERY REPORT ----------
def report(texts, E, mu, lab, XY, ents, L, Gi):
    """R6, R8, R10, R11. Two views per selected community: the query's (top hits by score) and the community's own (unbiased sample:
    medoid + spread picks + entities). Writes query_entities.png and query_results.toon."""
    N = len(texts); sizes = np.bincount(lab); n_show = n_per_community(sizes); rng = np.random.default_rng(1)
    rg = np.random.default_rng(0); pa, pb = rg.integers(0, N, 200000), rg.integers(0, N, 200000); ok = pa != pb
    refs = (np.sort(pair_cos(E, pa[ok], pb[ok])), np.sort(pair_cos(L, pa[ok], pb[ok])))
    df_all = doc_freq(ents, range(N)); Q = center(embed_queries(QUERIES), mu); toon, panels = [], []
    snip = lambda t: t if len(t) <= 160 else t[:157].rstrip() + "..."
    for qi, (qt, qv) in enumerate(zip(QUERIES, Q)):
        sim = E @ qv; hit = np.argsort(-sim)[:TOP_N]; hc = np.bincount(lab[hit], minlength=len(sizes))
        top = [c for c in np.argsort(-hc, kind="stable")[:TOP_K] if hc[c] >= MIN_HITS]; blocks, clus, cq, ce = [], [], [], []
        for k, c in enumerate(top):
            v = community_view(c, lab, E, L, Gi, refs, n_show, ents, df_all, rng); enr = hc[c] / TOP_N / (sizes[c] / N)
            qh = [(r + 1, float(sim[i]), i) for r, i in enumerate(hit) if lab[i] == c][:int(n_show[c])]
            clus.append([c, sizes[c], hc[c], round(float(enr), 1), len(v["smp"])])
            cq += [[c, "abc"[j], round(zz, 2), snip(texts[i])] for j, (i, zz) in enumerate(zip(v["picks"], v["z"]))]; ce += [[c, cl, t, n] for cl, t, n in v["ent"]]
            blocks.append((k, c, enr, v, qh))
        top10 = hit[:10]
        hrow = [[r + 1, lab[i], round(float(sim[i]), 3), snip(texts[i])] for r, i in enumerate(top10)]
        hent = [[r + 1, l, t] for r, i in enumerate(top10) for t, l in sorted(ents[i])]
        toon += [f"q{qi + 1}:", f"  query: {toon_str(qt)}"] + toon_table("hits", ["rank", "cluster", "cos", "text"], hrow, 2) \
              + toon_table("hit_entities", ["rank", "class", "entity"], hent, 2) \
              + toon_table("clusters", ["id", "size", "hits", "enrichment", "sample"], clus, 2) \
              + toon_table("community_view", ["cluster", "pick", "z", "text"], cq, 2) \
              + toon_table("community_entities", ["cluster", "class", "entity", "count"], ce, 2)
        panels.append((qt, sim, hit, top, blocks))
    # ---- figure: map on top, text (query view | community view | entities) per selected community below
    W = {"q": 46, "c": 52, "e": 38}; XQ, XC, XE = 0.0, 0.34, 0.72; text = []
    for qt, sim, hit, top, blocks in panels:
        rows = []
        for k, c, enr, v, qh in blocks:
            ql = [ln for r, cs, i in qh for ln in textwrap.wrap(f"{texts[i]}", W["q"], initial_indent=f"#{r} {cs:.2f}  ", subsequent_indent="      ")]
            cl = [ln for j, (i, zz) in enumerate(zip(v["picks"], v["z"])) for ln in textwrap.wrap(texts[i], W["c"], initial_indent=f"{'abc'[j]} z={zz:.2f}  ", subsequent_indent="        ")]
            el = [ln for cls, t, n in v["ent"] for ln in textwrap.wrap(f"{t} x{n}" if n > 1 else t, W["e"], initial_indent=f"{ABBR.get(cls, cls)}  ", subsequent_indent="      ")] or ["none in sample"]
            hn = sorted({f"{t}" for r, cs, i in qh for t, _ in ents[i]})
            if hn: el += textwrap.wrap(", ".join(hn[:6]), W["e"], initial_indent="in hits  ", subsequent_indent="         ")
            rows.append((k, c, enr, v, ql, cl, el))
        text.append(rows)
    NL = max(sum(2 + max(len(ql), len(cl), len(el)) + 1 for *_, ql, cl, el in rows) for rows in text) + 1
    fig = plt.figure(figsize=(12 * len(QUERIES), 12 + NL * 0.2), facecolor="#fcfcfb")
    gs = fig.add_gridspec(2, len(QUERIES), height_ratios=[12, NL * 0.2], hspace=0.01, wspace=0.03, left=0.005, right=0.995, top=0.955, bottom=0.005); fs = 9.5
    for qi, ((qt, sim, hit, top, blocks), rows) in enumerate(zip(panels, text)):
        ax = fig.add_subplot(gs[0, qi]); ax.set_facecolor("#fcfcfb"); ax.scatter(XY[:, 0], XY[:, 1], s=12, c=GREY, linewidths=0)
        for k, c, enr, v, *_ in rows: ax.scatter(XY[v["m"], 0], XY[v["m"], 1], s=34, c=HI[k], linewidths=0, alpha=0.85)
        ax.scatter(XY[hit, 0], XY[hit, 1], s=60, facecolors="none", edgecolors=INK, linewidths=0.8)
        for k, c, enr, v, *_ in rows: ax.scatter(XY[v["picks"][0], 0], XY[v["picks"][0], 1], s=420, marker="*", c=HI[k], edgecolors=INK, linewidths=1.2, zorder=5)
        ax.set_xticks([]); ax.set_yticks([]); [s_.set_visible(False) for s_ in ax.spines.values()]
        ax.set_title(f"Q{qi + 1}: {qt}\ncircled = top-{TOP_N} dense hits; colour = the {len(rows)} most-selected communities; star = medoid of the sample", fontsize=13, loc="left", color=INK)
        tx = fig.add_subplot(gs[1, qi]); tx.axis("off"); tx.set_xlim(0, 1); tx.set_ylim(NL, 0); y = 0.5
        for k, c, enr, v, ql, cl, el in rows:
            tx.text(0, y, f"■ community {c} · size {sizes[c]} · hits {hc_of(lab, hit, c)}/{TOP_N} · enrichment x{enr:.1f}", fontsize=fs + 1.5, fontweight="bold", color=HI[k], va="center", family="monospace"); y += 1
            for x0, ttl in ((XQ, "query view: top hits by score"), (XC, f"community view: sample of {len(v['smp'])}, medoid first"), (XE, "entities (sample) / in hits")):
                tx.text(x0, y, ttl, fontsize=fs, fontweight="bold", color=INK2, va="center", family="monospace")
            y += 1
            for x0, ls in ((XQ, ql), (XC, cl), (XE, el)):
                for j, ln in enumerate(ls): tx.text(x0, y + j, ln, fontsize=fs, color=INK, va="center", family="monospace")
            y += max(len(ql), len(cl), len(el)) + 1
    fig.savefig("query_entities.png", dpi=100); open("query_results.toon", "w").write("\n".join(toon) + "\n")


def hc_of(lab, hit, c): return int((lab[hit] == c).sum())


def embed_queries(qs):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(DENSE).encode(qs, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)


def main():
    texts = load_texts(); raw = embed_raw(texts); mu = raw.mean(0); E = center(raw, mu)
    idx, dist = knn(E); G = graph(idx, dist); L = lex_vectors(texts); Gi = G.multiply(graph(*knn(L))).tocsr(); Gi.eliminate_zeros()
    labs = [consensus(G, range(100 * r, 100 * r + SEEDS)) for r in range(3)]; lab = labs[0]
    print(f"{len(texts)} texts, {lab.max() + 1} communities, consensus seed-ARI {np.mean([ari(labs[a], labs[b]) for a in range(3) for b in range(a + 1, 3)]):.3f}")
    XY = umap.UMAP(n_components=2, metric="cosine", precomputed_knn=(idx, dist), random_state=7).fit_transform(E)
    report(texts, E, mu, lab, XY, entities(texts), L, Gi); print("wrote query_entities.png, query_results.toon")


if __name__ == "__main__":
    main()
