"""
chunkgraph.py — dual-space retrieval graph over arbitrary text.

STAGED PIPELINE
| stage    | mechanism                                                            | tag        |
|----------|----------------------------------------------------------------------|------------|
| CHUNK    | paragraph split; sliding-window fallback; doc_id retained (R8)       | [mandatory]|
| PHRASE   | Dunning-LLR report + NPMI Phrases merged into tokens pre-BM25 (R9)   | [opt: R9]  |
| SPARSE   | BM25-weighted CSR, L2 rows, blockwise X@X.T, in-loop threshold (R10) | [mandatory]|
| DENSE    | pluggable embed_fn (default: local MiniLM mean-pool); cosine sim     | [opt: R5]  |
| NORMAL   | Box-Cox per sim distribution (lambda stored); estimator-pair gate    | [mandatory]|
| EDGES    | k-sigma tail cut where post-kurt<=KURT_OK, else budget-match (R2)    | [mandatory]|
| BACKBONE | per-node top-KNN neighbors unioned in, both spaces (R7)              | [mandatory]|
| FUSE     | union graph + provenance {sparse,dense,both}; BC-z edge strength,    | [mandatory]|
|          | rank-blend fallback (R3)                                             |            |
| COMMUNITY| Louvain on union (weight=strength); tf*idf keywords; blend medoids   | [opt]      |
| SERVE    | query(): validated anchors -> H-hop frontier -> GIST-walk            | [mandatory]|
| DRAW     | spring layout, provenance-colored edges, keyword labels              | [opt]      |
| EXPORT   | edges(): rows w/ provenance, strength, raw sims, bitemporal cols     | [opt]      |

GUARDS (EARS)
R1 WHEN an anchor is selected, the matched query term SHALL appear in that
   chunk's top DISC_M tf*idf terms (anchor validation; incidental-tf rejected).
R2 IF post-Box-Cox kurtosis > KURT_OK for a space, THEN its edge cut SHALL be
   budget-matched to the other space's edge count instead of k-sigma.
R3 IF Box-Cox fails (degenerate distribution), THEN blend strength SHALL fall
   back to mean rank across spaces.
R4 The GIST-walk SHALL accept a threshold only if the greedy fills >= MINK
   slots (min-cardinality feasibility).
R7 WHEN edges are built, each node SHALL retain its KNN nearest neighbors in
   each space regardless of significance; significance ranks edges, the
   backbone guarantees connectivity (a k-sigma cut alone leaves isolates).
R5 WHERE embed_fn is None and no model_dir is supplied, the system runs
   sparse-only: dense stage, fusion tiers, and blend degrade gracefully.
R8 WHEN chunks are produced, each SHALL carry its source doc_id so every
   derived edge can name the documents it came from (retraction by source).
R9 WHERE phrases=True and gensim is importable, multiword phrases SHALL be
   merged into single tokens BEFORE BM25 so they become vocab terms.
R10 The chunk-chunk product SHALL be thresholded inside the block loop; a
   materialized n x n dense product exhausts memory at ~18k chunks.
R11 WHEN edges are exported, each row SHALL carry valid_from/valid_to/
   ingested_at. Edges are closed (valid_to set), never overwritten.
R12 WHERE vocab is supplied, the CSR SHALL be built over those terms only.
   Callers pass a selected vocabulary (e.g. salient_grams.select_vocab_unified
   -> result['terms']); idf and qterms are recomputed over the restriction so
   scoring stays internally consistent.
R6 WHEN both estimators (median/1.4826*MAD vs mean/std) diverge > DIV_WARN in
   BC space, fit() SHALL record a warning in self.diagnostics.

CLOSED (evidence from Brown-corpus validation, this session)
- Raw MAD/mean cuts on untransformed sims: rejected (skew +1.4..+2.8 broke both).
- Intersection as traversal graph: rejected (starves walks); kept as tier only.
- Query-side WordNet expansion: rejected (Voorhees-style drift observed).
- Rank-blend as primary strength: superseded by BC-z (rho .90, homophily .95>.93).
- Significance-only edge sets: rejected for layout/traversal (23% isolates,
  655 components at 1821 nodes); backbone union required.
- Box-Cox scores for LAYOUT: rejected; positions use RAW sims (normalized
  scores still drive thresholds, tiers, strengths, walk distances).
- Dense n x n BM25 loop in fit(): rejected (O(n^2) python calls; 45M pairs
  materialized at 18k chunks -> OOM). Replaced by CSR blockwise product.
- A second graph engine for 1-2 hop traversal: rejected. Edge rows in the
  existing store cover it; an engine earns its place at deep traversal.

Preconditions: >= ~50 chunks for stable distributions. Failure modes: tiny
corpora make Box-Cox unstable (R3 path); embed_fn absent disables dense (R5).
"""
import math, re, warnings
import numpy as np
import scipy.sparse as sp
from collections import Counter
from datetime import datetime, timezone
from scipy import stats

DISC_M, MINK, KURT_OK, DIV_WARN, KNN = 25, 3, 0.5, 0.15, 2
SIM_FLOOR, BLOCK = 0.02, 512   # R10: in-loop threshold, block rows
_STOP = set("""the of and to a in that is was he for it with as his on be at by i
this had not are but from or have an they which one you were her all she there
would their we him been has when who will more no if out so said what up its
about into than them can only other new some could time these two may then do
first any my now such like our over man me even most made after also did many
before must through back years where much your way well down should because""".split())

def _tok(text):
    return [w for w in re.findall(r"[a-z]+", text.lower()) if w not in _STOP and len(w) > 2]

def _chunk(doc, target=120, max_len=200):
    """Require: doc str. Guarantee: list of chunk strings, paragraph-first."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", doc) if p.strip()]
    out = []
    for p in paras:
        words = p.split()
        if len(words) <= max_len:
            out.append(p)
        else:  # sliding window, 50% overlap
            step = target // 2
            out += [" ".join(words[i:i+target]) for i in range(0, len(words)-step, step)]
    return out

def default_embed_fn(model_dir):
    """Guarantee: texts -> L2-normalized np.ndarray. Requires torch+transformers."""
    import torch, torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModel.from_pretrained(model_dir); mdl.eval()
    def embed(texts, bs=32):
        outs = []
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                enc = tok(texts[i:i+bs], padding=True, truncation=True,
                          max_length=256, return_tensors="pt")
                h = mdl(**enc).last_hidden_state
                m = enc.attention_mask.unsqueeze(-1)
                outs.append(F.normalize((h*m).sum(1)/m.sum(1), dim=1))
        return torch.cat(outs).numpy()
    return embed

class ChunkGraph:
    """Require: docs = list[str]. Guarantee after fit(): union graph, edge
    strengths, query() and communities() operational. Maintain: all thresholds
    derived from data (lambda, k*sigma) are stored in self.diagnostics."""

    def __init__(self, embed_fn=None, model_dir=None, k_sigma=2.0,
                 query_terms=20, hops=2, k_expand=6, lam_div=8.0, eps=0.15,
                 phrases=True, vocab=None):
        if embed_fn is None and model_dir is not None:
            embed_fn = default_embed_fn(model_dir)
        self.embed_fn = embed_fn                                    # R5
        self.k_sigma, self.M = k_sigma, query_terms
        self.H, self.K, self.LAM, self.EPS = hops, k_expand, lam_div, eps
        self.phrases = phrases
        self.vocab = set(vocab) if vocab else None            # R12
        self.diagnostics = {}
        self.ingested_at = datetime.now(timezone.utc).isoformat()

    # ---------- fit ----------
    def fit(self, docs, doc_ids=None):
        """Require: docs list[str]. Guarantee: graph fitted; self.doc_id[i]
        names the source document of chunk i (R8)."""
        doc_ids = doc_ids or [f"doc{i}" for i in range(len(docs))]
        self.chunks, self.doc_id = [], []
        for did, d in zip(doc_ids, docs):
            cs = _chunk(d)
            self.chunks += cs; self.doc_id += [did]*len(cs)
        raw_tok = [_tok(c) for c in self.chunks]
        self.docs_tok = self._merge_phrases(raw_tok) if self.phrases else raw_tok   # R9
        n = self.n = len(self.chunks)
        assert n >= 10, "need >=10 chunks"
        if self.vocab is not None:                             # R12
            self.docs_tok = [[t for t in d if t in self.vocab] for d in self.docs_tok]
            self.diagnostics["vocab"] = {"mode": "restricted", "size": len(self.vocab)}
        df = Counter(t for d in self.docs_tok for t in set(d))
        self.idf = {t: math.log(1 + (n - c + .5)/(c + .5)) for t, c in df.items()}
        self.tfs = [Counter(d) for d in self.docs_tok]
        self.avgdl = sum(map(len, self.docs_tok))/n
        self.qterms = [ [t for t,_ in sorted(tf.items(),
                         key=lambda kv: -kv[1]*self.idf.get(kv[0],0))[:self.M]]
                        for tf in self.tfs ]
        self.disc = [set(q[:DISC_M]) for q in self.qterms]
        self.sim_sparse = self._sparse_sim()                                        # R10
        self.sim_dense = None
        if self.embed_fn is not None:
            E = self.embed_fn(self.chunks)
            self.E = E/np.linalg.norm(E, axis=1, keepdims=True)
            self.sim_dense = self.E @ self.E.T; np.fill_diagonal(self.sim_dense, 0)
        self._edges_and_strength()
        return self

    def _merge_phrases(self, toks):
        """Guarantee: multiword phrases become single tokens (R9). Reports the
        Dunning-LLR top bigrams in diagnostics; NPMI does the merging."""
        try:
            from gensim.models.phrases import Phrases, Phraser
            from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
        except ImportError:
            self.diagnostics["phrases"] = {"mode": "unavailable"}
            return toks
        try:
            f = BigramCollocationFinder.from_documents(toks); f.apply_freq_filter(5)
            llr = [' '.join(b) for b in f.nbest(BigramAssocMeasures.likelihood_ratio, 20)]
        except Exception:
            llr = []
        out = [Phraser(Phrases(toks, min_count=5, threshold=0.4, scoring='npmi'))[t]
               for t in toks]
        self.diagnostics["phrases"] = {
            "mode": "npmi", "llr_top": llr[:10],
            "merged": len({t for d in out for t in d if '_' in t})}
        return out

    def _sparse_sim(self):
        """Guarantee: n x n sim from BM25-weighted L2 rows. Maintain: the block
        product is thresholded in-loop, never materialized whole (R10)."""
        vocab, rows, cols, vals = {}, [], [], []
        K1, B = 1.5, .75
        for i, d in enumerate(self.docs_tok):
            for t, f in Counter(d).items():
                if t not in self.idf: continue
                j = vocab.setdefault(t, len(vocab))
                rows.append(i); cols.append(j)
                vals.append(self.idf[t]*f*(K1+1)/(f + K1*(1-B+B*len(d)/self.avgdl)))
        X = sp.csr_matrix((vals, (rows, cols)), shape=(self.n, max(len(vocab), 1)))
        nr = np.sqrt(X.multiply(X).sum(1)); nr[nr == 0] = 1
        X = sp.csr_matrix(X.multiply(1/nr))
        S = np.zeros((self.n, self.n))
        for i0 in range(0, self.n, BLOCK):
            blk = (X[i0:i0+BLOCK] @ X.T).toarray()
            blk[blk < SIM_FLOOR] = 0                                   # R10
            S[i0:i0+blk.shape[0]] = blk
        np.fill_diagonal(S, 0)
        return np.maximum(S, S.T)

    def bm25(self, terms, j, K1=1.5, B=.75):
        tf, dl, s = self.tfs[j], len(self.docs_tok[j]), 0.0
        for t in terms:
            f = tf.get(t, 0)
            if f: s += self.idf[t]*f*(K1+1)/(f + K1*(1-B+B*dl/self.avgdl))
        return s

    # ---------- NORMAL + EDGES + FUSE ----------
    def _bc(self, x):
        try:
            y, lam = stats.boxcox(x - x.min() + 1e-3)
            return y, lam, stats.kurtosis(y)
        except Exception:
            return None, None, None                                 # R3

    def _cut(self, tri_vals, name):
        """Maintain: Box-Cox is fitted on NONZERO sims only. The in-loop floor
        (R10) zeroes most pairs; including them makes the fit a zero-inflation
        artifact (measured kurt 1.61 -> forced budget path)."""
        nz = tri_vals > 0
        y_nz, lam, kurt = self._bc(tri_vals[nz]) if nz.sum() >= 20 else (None, None, None)
        d = self.diagnostics.setdefault(name, {})
        if y_nz is None:
            d["mode"] = "rank-fallback"; return None, None
        y = np.full(tri_vals.shape, y_nz.min() - 1.0); y[nz] = y_nz
        d["nonzero_frac"] = float(nz.mean())
        med, mad = np.median(y_nz), stats.median_abs_deviation(y_nz)
        mu, sd = y_nz.mean(), y_nz.std()
        div = abs(med-mu)/sd + abs(1.4826*mad - sd)/sd
        d.update(dict(mode="boxcox", lam=lam, post_kurt=kurt, divergence=div))
        if div > DIV_WARN:
            d["warning"] = "estimator divergence: normalization imperfect"  # R6
        z = (y - mu)/sd
        mask = None if kurt > KURT_OK else ((y > mu + self.k_sigma*sd) & nz)  # R2 defer
        return z, mask

    @staticmethod
    def _backbone(sim, k=KNN):
        """Guarantee: boolean adj where every node keeps its k nearest (R7)."""
        n = sim.shape[0]
        A = np.zeros((n, n), bool)
        kk = min(k, n-1)
        for i in range(n):
            for j in np.argpartition(-sim[i], kk)[:kk+1]:
                if j != i: A[i, j] = True
        return A | A.T

    def _edges_and_strength(self):
        n, tri = self.n, np.triu_indices(self.n, 1)
        z_sp, m_sp = self._cut(self.sim_sparse[tri], "sparse")
        A_sp = np.zeros((n, n), bool)
        if m_sp is None:   # kurtosis too high or bc failed -> budget by z rank
            budget = max(n, int(2*n))
            m_sp = np.argsort(-(z_sp if z_sp is not None else self.sim_sparse[tri]))[:budget]
            sel = np.zeros(len(tri[0]), bool); sel[m_sp] = True; m_sp = sel
            self.diagnostics["sparse"]["mode"] += "+budget"
        A_sp[tri[0][m_sp], tri[1][m_sp]] = True; A_sp |= A_sp.T
        A_sp |= self._backbone(self.sim_sparse)                       # R7
        self.A_sparse = A_sp

        if self.sim_dense is not None:
            z_de, m_de = self._cut(self.sim_dense[tri], "dense")
            A_de = np.zeros((n, n), bool)
            if m_de is None or m_de.sum() > 3*A_sp.sum()/2:          # R2
                budget = A_sp.sum()//2
                order = np.argsort(-(z_de if z_de is not None else self.sim_dense[tri]))[:budget]
                sel = np.zeros(len(tri[0]), bool); sel[order] = True; m_de = sel
                self.diagnostics["dense"]["mode"] = self.diagnostics["dense"].get("mode","") + "+budget-matched"
            A_de[tri[0][m_de], tri[1][m_de]] = True; A_de |= A_de.T
            A_de |= self._backbone(self.sim_dense)                    # R7
            self.A_dense = A_de
            self.A = A_sp | A_de
            self.provenance = np.where(A_sp & A_de, 3, np.where(A_de, 2, np.where(A_sp, 1, 0)))
            if z_sp is not None and z_de is not None:                # strength
                self.strength = np.zeros((n, n))
                self.strength[tri] = (z_sp + z_de)/2
                self.strength += self.strength.T
                self.blend_mode = "bc-z"
            else:                                                     # R3
                r = lambda S: np.argsort(np.argsort(S, 1), 1)/(n-1)
                self.strength = (r(self.sim_sparse) + r(self.sim_dense))/2
                self.blend_mode = "rank"
            self.D = 1 - (self.strength - self.strength.min())/(np.ptp(self.strength) + 1e-9)
        else:                                                         # R5 sparse-only
            self.A_dense, self.A = None, A_sp
            self.provenance = A_sp.astype(int)
            self.strength = self.sim_sparse.copy(); self.blend_mode = "sparse-only"
            self.D = 1 - self.sim_sparse
        np.fill_diagonal(self.D, 0)

    # ---------- SERVE ----------
    def query(self, q, k_anchor=2):
        qt = _tok(q); qset = set(qt)
        scores = np.array([self.bm25(qt, j) for j in range(self.n)])
        order = np.argsort(-scores)
        anchors = [j for j in order if scores[j] > 0 and (qset & self.disc[j])][:k_anchor]  # R1
        if not anchors:
            return dict(anchors=[], expansion=[], note="no valid anchor (R1)")
        seen, cur = set(anchors), set(anchors)
        for _ in range(self.H):
            nxt = set()
            for v in cur: nxt.update(np.where(self.A[v])[0])
            nxt -= seen; seen |= nxt; cur = nxt
        cand = list(seen - set(anchors))
        sel = self._gist_walk(anchors, cand, scores)
        return dict(anchors=anchors, expansion=sel, scores=scores,
                    texts=[self.chunks[j][:160] for j in anchors + sel])

    def _gist_walk(self, seeds, cand, scores):
        if not cand: return []
        dmax = max((self.D[a, b] for a in cand for b in cand), default=1) or 1
        ths, t = [0.0], self.EPS*dmax/2
        while t <= dmax: ths.append(t); t *= (1+self.EPS)
        best, bestf = [], -1
        for d in ths:
            sel, pool = [], list(cand)
            while len(sel) < self.K and pool:
                v = max(pool, key=lambda x: scores[x])
                sel.append(v)
                pool = [x for x in pool if x != v and self.D[x, v] >= d
                        and all(self.D[x, s] >= d for s in seeds)]
            if len(sel) < min(MINK, len(cand)): continue              # R4
            dv = min((self.D[a, b] for a in sel for b in sel if a != b), default=dmax)
            f = sum(scores[v] for v in sel) + self.LAM*dv
            if f > bestf: bestf, best = f, sel
        return best

    # ---------- COMMUNITY ----------
    def communities(self, min_size=5):
        import networkx as nx, community as louvain
        Gx = nx.Graph(); Gx.add_nodes_from(range(self.n))
        for a, b in zip(*np.where(np.triu(self.A, 1))):
            Gx.add_edge(int(a), int(b), weight=float(max(self.strength[a, b], 1e-6)))
        part = louvain.best_partition(Gx, weight="weight", random_state=7)
        out, groups = [], {}
        for v, c in part.items(): groups.setdefault(c, []).append(v)
        for c, vs in groups.items():
            if len(vs) < min_size: continue
            tc = Counter()
            for v in vs:
                for t, f in self.tfs[v].items(): tc[t] += f
            kws = [t for t,_ in sorted(tc.items(), key=lambda kv: -kv[1]*self.idf.get(kv[0],0))[:5]]
            sub = np.ix_(vs, vs)
            med = vs[int(np.argmin(self.D[sub].sum(1)))]
            out.append(dict(id=c, members=vs, keywords=kws, medoid=med,
                            medoid_text=self.chunks[med][:160]))
        self._part = part
        return sorted(out, key=lambda d: -len(d["members"]))

    # ---------- EXPORT ----------
    def edges(self, valid_from=None):
        """Guarantee: one row per undirected edge, carrying provenance, blended
        strength, both raw sims, source documents, and bitemporal columns (R11).
        Maps 1:1 onto an edge table; valid_to is None until superseded."""
        PROV = {1: "sparse", 2: "dense", 3: "both"}
        out = []
        for a, b in zip(*np.where(np.triu(self.A, 1))):
            a, b = int(a), int(b)
            out.append(dict(
                src=a, dst=b,
                src_doc=self.doc_id[a], dst_doc=self.doc_id[b],
                provenance=PROV.get(int(self.provenance[a, b]), "sparse"),
                strength=float(self.strength[a, b]),
                sim_sparse=float(self.sim_sparse[a, b]),
                sim_dense=float(self.sim_dense[a, b]) if self.sim_dense is not None else None,
                valid_from=valid_from or self.ingested_at,
                valid_to=None,
                ingested_at=self.ingested_at))
        return out

    # ---------- DRAW ----------
    def draw(self, path, min_size=5):
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt, networkx as nx
        comms = self.communities(min_size)
        Gx = nx.Graph(); Gx.add_nodes_from(range(self.n))
        for a, b in zip(*np.where(np.triu(self.A, 1))):
            raw = (self.sim_dense[a, b] if self.sim_dense is not None
                   else self.sim_sparse[a, b])
            Gx.add_edge(int(a), int(b), weight=float(max(raw, .05)))   # nominal scale
        pos = nx.spring_layout(Gx, weight="weight", k=0.9, iterations=200, seed=7)
        fig, ax = plt.subplots(figsize=(15, 11), dpi=110)
        cmap = plt.get_cmap("tab20")
        for a, b in Gx.edges():
            p = self.provenance[a, b]
            col, w, al = {3: ("#d62728", 1.5, .5), 2: ("#1f77b4", .7, .3)}.get(p, ("#999999", .7, .25))
            ax.plot(*zip(pos[a], pos[b]), color=col, lw=w, alpha=al, zorder=1)
        incomm = set()
        for i, cd in enumerate(comms):
            vs = cd["members"]; incomm |= set(vs)
            xs, ys = zip(*(pos[v] for v in vs))
            ax.scatter(xs, ys, s=60, color=cmap(i % 20), edgecolors="black", lw=.4, zorder=3)
            ax.scatter(*pos[cd["medoid"]], s=320, marker="*", color=cmap(i % 20),
                       edgecolors="black", lw=1.1, zorder=4)
            ax.annotate(" / ".join(cd["keywords"][:4]), (np.mean(xs), np.mean(ys)),
                        fontsize=10, fontweight="bold", ha="center", zorder=5,
                        bbox=dict(boxstyle="round,pad=.3", fc="white",
                                  ec=cmap(i % 20), lw=1.6, alpha=.92))
        rest = [v for v in Gx if v not in incomm]
        ax.scatter([pos[v][0] for v in rest], [pos[v][1] for v in rest],
                   s=24, c="#dddddd", edgecolors="#aaaaaa", lw=.4, zorder=2)
        ax.set_title(f"ChunkGraph: {self.n} chunks | blend={self.blend_mode} | "
                     f"edges={self.A.sum()//2}", fontsize=12)
        ax.axis("off"); plt.tight_layout()
        plt.savefig(path, bbox_inches="tight"); plt.close()
        return path
