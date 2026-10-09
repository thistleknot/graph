#!/usr/bin/env python3
"""
graph3d.py — build an explorable 3D chunk graph from a document set.

STAGE TABLE
| stage   | mechanism                                                            | tag         |
|---------|----------------------------------------------------------------------|-------------|
| SAMPLE  | N_DOCS Brown files, strided for genre spread (swap for your corpus)  | [mandatory] |
| DERIVE  | chunk_size = para_median + 1.4826*MAD; overlap = 2*sent_median,      | [mandatory] |
|         | capped 0.45*chunk_size — no hand-set constants                       |             |
| SPLIT   | RecursiveCharacterTextSplitter                                       | [mandatory] |
| VOCAB   | salient_grams.select_vocab_unified: df band + keyness + anomaly +    | [opt: R6]   |
|         | GAMM spline + MMR; tokens restricted to selection before BM25        |             |
| NGRAM   | NLTK BigramCollocationFinder likelihood_ratio (Dunning 1993) for     | [mandatory] |
|         | reporting + gensim Phrases NPMI merged into tokens pre-BM25 (R1)     |             |
| SPARSE  | BM25-weighted CSR, L2 rows, blockwise X@X.T with IN-LOOP threshold   | [mandatory] |
| NORMAL  | Box-Cox on nonzero sims; k-sigma cut, k raised if post-kurt > .5     | [mandatory] |
| DENSE   | model2vec static vectors; budget-matched to sparse edge count (R2)   | [mandatory] |
| BACKBONE| per-node top-KNN neighbors in each space, unioned in (R3)            | [mandatory] |
| FUSE    | union graph; both-space tier; Louvain; tf*idf community keywords     | [mandatory] |
| DRAW3D  | spring_layout(dim=3) weighted by RAW sims (R4) -> plotly HTML        | [mandatory] |

GUARDS
R1 Phrase merge happens BEFORE BM25 so multiword entities are vocab terms.
R2 Dense edge budget SHALL equal the sparse edge count (degree-match).
R3 WHEN edges are built, every node SHALL keep its KNN nearest neighbors in
   each space regardless of significance. Significance ranks edges; the
   backbone guarantees connectivity.
R4 Layout weights SHALL be raw similarities. Box-Cox scores drive thresholds,
   tiers, and traversal distances — never node positions.
R6 WHERE USE_VOCAB is set, tokens SHALL be restricted to the selected
   vocabulary before the BM25 CSR is built. Measured on Brown chunks:
   3832 -> 1758 terms, edges +2%, homophily flat, nonzero density
   0.163 -> 0.198, chunk coverage 1.00 (no chunk loses all its terms).
R7 The chars/row waterline does NOT bind at chunk granularity
   (contribution = len(t)*df/N; measured 191 vs a 315 budget), so MMR
   ranks but does not exclude. Size the vocabulary with the df band and
   spline, not the waterline, when rows are chunks rather than documents.
R5 IF the block product is materialized whole, memory blows up at ~18k chunks;
   the similarity threshold SHALL be applied inside the block loop.

CLOSED (measured this session, do not re-litigate without new data)
- include_plotlyjs='cdn': fails offline/sandboxed. Inline the library.
- Significance-only edges: 23% isolates, 655 components, unreadable hairball.
- Box-Cox-scaled layout: rejected (R4).
- chars/row waterline as the vocabulary size dial for chunks: rejected
  (never binds; 315/2000/8000 all returned the identical 1758 terms).

Preconditions: nltk brown+punkt, gensim, langchain-text-splitters, model2vec,
networkx, python-louvain, plotly. Failure modes: <50 chunks destabilizes Box-Cox.
"""
import re, math, numpy as np, scipy.sparse as sp
from collections import Counter
from scipy import stats
from nltk.corpus import brown
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
from gensim.models.phrases import Phrases, Phraser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from model2vec import StaticModel
import networkx as nx, community as louvain
import plotly.graph_objects as go

N_DOCS, KNN_SP, KNN_DE, K_SIGMA = 50, 2, 3, 2.0
USE_VOCAB = True          # R6: salient-term restriction before BM25
M2V_DIR = '/home/claude/m2v-minilm'
OUT = '/mnt/user-data/outputs/graph3d.html'
STOP = set("""the of and to a in that is was he for it with as his on be at by i this had not are but
from or have an they which one you were her all she there would their we him been has when will who
more no if out so said what up its about into than them can only other new some could time may""".split())

# ---------- SAMPLE ----------
fids = brown.fileids()[::10][:N_DOCS]
docs, pl, sl = [], [], []
for fid in fids:
    paras = ['\n'.join(' '.join(s) for s in p) for p in brown.paras(fileids=fid)]
    docs.append('\n\n'.join(paras))
    pl += [len(p) for p in paras]
    sl += [len(' '.join(s)) for p in brown.paras(fileids=fid) for s in p]

# ---------- DERIVE + SPLIT ----------
CS = int(np.median(pl) + 1.4826*stats.median_abs_deviation(pl))
OV = min(int(2*np.median(sl)), int(.45*CS))
chunks = [c for d in docs
          for c in RecursiveCharacterTextSplitter(chunk_size=CS, chunk_overlap=OV).split_text(d)]
N = len(chunks)
print(f"{len(docs)} docs -> derived chunk:overlap = {CS}:{OV} -> {N} chunks")

# ---------- NGRAM (R1) ----------
base_tok = lambda t: [w for w in re.findall(r"[a-z]+", t.lower()) if len(w) > 2 and w not in STOP]
sents = [base_tok(c) for c in chunks]
finder = BigramCollocationFinder.from_documents(sents); finder.apply_freq_filter(5)
llr = finder.nbest(BigramAssocMeasures.likelihood_ratio, 60)
toks = [Phraser(Phrases(sents, min_count=5, threshold=0.4, scoring='npmi'))[s] for s in sents]
print(f"Dunning-LLR: {[' '.join(b) for b in llr[:8]]}")

# ---------- VOCAB (R6) ----------
if USE_VOCAB:
    import salient_grams as sg
    sel = sg.select_vocab_unified(chunks)
    keep = set(sel["terms"])
    kept_toks = [[t for t in d if t in keep or '_' in t] for d in toks]   # phrases survive
    cov = sum(1 for d in kept_toks if d)/len(kept_toks)
    print(f"vocab: {sel['n_eligible']} eligible -> {sel['n_selected']} selected "
          f"(chars/row {sel['chars_per_row']:.0f}), chunk coverage {cov:.2f}")
    if cov > 0.98:                       # guard: never strand chunks
        toks = kept_toks
    else:
        print("  coverage too low, keeping full vocabulary")
print(f"NPMI phrases in vocab: {len({t for d in toks for t in d if '_' in t})}")

# ---------- SPARSE (R5) ----------
df = Counter(t for d in toks for t in set(d))
idf = {t: math.log(1 + (N-c+.5)/(c+.5)) for t, c in df.items() if c >= 2}
avgdl = np.mean([len(d) for d in toks]); K1, B = 1.5, .75
rows, cols, vals, vocab = [], [], [], {}
for i, d in enumerate(toks):
    for t, f in Counter(t for t in d if t in idf).items():
        j = vocab.setdefault(t, len(vocab))
        rows.append(i); cols.append(j)
        vals.append(idf[t]*f*(K1+1)/(f + K1*(1-B+B*len(d)/avgdl)))
X = sp.csr_matrix((vals, (rows, cols)), shape=(N, len(vocab)))
X = sp.csr_matrix(X.multiply(1/np.sqrt(X.multiply(X).sum(1))))
pairs = {}
for i0 in range(0, N, 400):
    Sb = (X[i0:i0+400] @ X.T).toarray()
    for r in range(Sb.shape[0]):
        gi = i0 + r
        for j in np.where(Sb[r] > .05)[0]:
            if j > gi: pairs[(gi, int(j))] = float(Sb[r, j])

# ---------- NORMAL ----------
sv = np.array(list(pairs.values()))
y, lam = stats.boxcox(sv + 1e-6); kurt = stats.kurtosis(y)
ks = K_SIGMA + (0.5 if kurt > .5 else 0)
cut = y.mean() + ks*y.std()
E_sp = {p for p, yy in zip(pairs, y) if yy > cut}
print(f"sparse: lam={lam:.2f} post_kurt={kurt:.2f} -> {ks}sigma -> {len(E_sp)} significant edges")

# ---------- DENSE (R2) ----------
E = StaticModel.from_pretrained(M2V_DIR).encode(chunks)
E = E/np.linalg.norm(E, axis=1, keepdims=True)
dcand = {}
for i0 in range(0, N, 512):
    S = E[i0:i0+512] @ E.T
    for r in range(S.shape[0]):
        S[r, i0+r] = -1
        for j in np.argpartition(-S[r], KNN_DE)[:KNN_DE]:
            k = (min(i0+r, int(j)), max(i0+r, int(j)))
            dcand[k] = max(dcand.get(k, -1), float(S[r, j]))
E_de = set(p for p, _ in sorted(dcand.items(), key=lambda kv: -kv[1])[:len(E_sp)])

# ---------- BACKBONE (R3) ----------
adj = {}
for (a, b), s in pairs.items():
    adj.setdefault(a, []).append((s, b)); adj.setdefault(b, []).append((s, a))
back = {(min(v, u), max(v, u)): s
        for v, lst in adj.items() for s, u in sorted(lst, reverse=True)[:KNN_SP]}
E_union = set(E_sp) | set(E_de) | set(back) | set(dcand)
E_both = (set(E_sp) | set(back)) & (set(E_de) | set(dcand))
W = {**dcand, **pairs}                                   # raw scales for layout (R4)

# ---------- FUSE ----------
G = nx.Graph()
for (a, b) in E_union: G.add_edge(a, b, weight=max(W.get((a, b), .05), .05))
print(f"graph: nodes={G.number_of_nodes()} edges={G.number_of_edges()} "
      f"avg_deg={2*G.number_of_edges()/G.number_of_nodes():.2f} "
      f"components={nx.number_connected_components(G)} both-tier={len(E_both)/len(E_union):.0%}")
part = louvain.best_partition(G, weight='weight', random_state=7)
groups = {}
for v, c in part.items(): groups.setdefault(c, []).append(v)
big = sorted([vs for vs in groups.values() if len(vs) >= 15], key=len, reverse=True)[:16]
tfs = [Counter(d) for d in toks]
labels = []
for vs in big:
    tc = Counter()
    for v in vs:
        for t, f in tfs[v].items(): tc[t] += f
    labels.append(' / '.join(t.replace('_', ' ') for t, _ in
                  sorted(tc.items(), key=lambda kv: -kv[1]*idf.get(kv[0], 0))[:3]))
print("communities:", labels)

# ---------- DRAW3D (R4) ----------
pos = nx.spring_layout(G, dim=3, weight='weight', k=0.9, iterations=200, seed=7)
nodes = sorted(G.nodes()); ix = {v: i for i, v in enumerate(nodes)}
P3 = np.array([pos[v] for v in nodes])
cmap = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f',
        '#bcbd22','#17becf','#aec7e8','#ffbb78','#98df8a','#ff9896','#c5b0d5','#c49c94']
col = ['#d9d9d9']*len(nodes)
for i, vs in enumerate(big):
    for v in vs: col[ix[v]] = cmap[i % 16]
def etrace(edges, c, w, o):
    xs, ys, zs = [], [], []
    for a, b in edges:
        xs += [P3[ix[a],0], P3[ix[b],0], None]
        ys += [P3[ix[a],1], P3[ix[b],1], None]
        zs += [P3[ix[a],2], P3[ix[b],2], None]
    return go.Scatter3d(x=xs, y=ys, z=zs, mode='lines',
                        line=dict(color=c, width=w), opacity=o, hoverinfo='none')
fig = go.Figure([
    etrace([e for e in E_union if e not in E_both], '#b0b0b0', 1, .18),
    etrace(E_both, '#d62728', 2.5, .55),
    go.Scatter3d(x=P3[:,0], y=P3[:,1], z=P3[:,2], mode='markers',
                 marker=dict(size=4, color=col, line=dict(width=.3, color='#333')),
                 text=[chunks[v][:120].replace('\n', ' ') for v in nodes], hoverinfo='text'),
    go.Scatter3d(x=[np.mean([P3[ix[v],0] for v in vs]) for vs in big],
                 y=[np.mean([P3[ix[v],1] for v in vs]) for vs in big],
                 z=[np.mean([P3[ix[v],2] for v in vs]) for vs in big],
                 mode='text', text=labels, textfont=dict(size=12, color='#000'), hoverinfo='none')])
fig.update_layout(title=f'{len(docs)} docs — {G.number_of_nodes()} chunks, {G.number_of_edges()} edges '
                        f'(backbone + significant; red = both-space)'
                        f'{" | salient vocab" if USE_VOCAB else ""}. Pinch/scroll zoom, drag orbit',
                  scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
                             dragmode='orbit'),
                  showlegend=False, margin=dict(l=0, r=0, t=45, b=0))
fig.write_html(OUT, include_plotlyjs=True,
               config={'scrollZoom': True, 'displayModeBar': True, 'responsive': True})
html = open(OUT).read().replace('<head>',
    '<head><meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=yes">'
    '<style>html,body{margin:0;height:100%;touch-action:none;}canvas{touch-action:none;}</style>')
open(OUT, 'w').write(html)
print(f"saved {OUT}")
