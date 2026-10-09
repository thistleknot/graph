"""src/term_salience.py -- what a term does to a chunk in the sparse space and in the dense space, side by side.

NO GOVERNING SPEC. Basis: operator question 2026-10-04 ("how would we derive term salience with bpe tokenizer approach and
sparsevec ... isolate term influence ... subtractions between embeddings") and the approved plan
C:\\Users\\user\\.claude\\plans\\jolly-soaring-moler.md. Tasks: playbook.md T134, T136, T137, T138, T139. The measurements these
functions reproduce were first run as scratch (.tmp/term_probe*.py); their numbers are in the ledger lessons.

    SPARSE   a BM25 term's share of one lexical edge is exact: x_ik * x_jk / cos(i, j)          TS1
    COARSE   a term's spread over the communities (entropy) and its best community (G2)          TS2
    DENSE    delta_t(d) = e(d) - e(d without t): the term's influence on one chunk               TS3
    GATE     is delta_t a direction shared across chunks, or context-dependent noise?            TS4
    SURROGATE a token's share of the mean-pooled vector, one forward pass, no re-encoding        TS5
    SUBTRACT e*(t) = mean of delta_t over training chunks, tried on held-out chunks              TS6

Guards (EARS):

TS1  The share a term carries of an edge SHALL be x_ik * x_jk / cos(i, j) over L2-normalised BM25 rows, so the shares of every
     term of an edge sum to 1; an edge with cos <= 0 gives no share.
TS2  A term SHALL be assessed only when it holds at least `min_df` chunks, and over-representation (not under-) decides its best
     community.
TS3  Removing a term SHALL delete every whole-word, case-insensitive occurrence, then collapse whitespace; a chunk that does not
     contain the term SHALL NOT yield a pair.
TS4  The stability gate SHALL compare same-term delta pairs with mismatched-term pairs; it reports both medians with n beside each,
     and never a verdict without them.
TS5  The token-share vector SHALL be the sum of the term's token vectors over ALL tokens of the chunk (n of the mean), so it is
     the term's part of the pooled vector before normalisation.
TS6  Subtraction gain SHALL be reported against a wrong-term control; a gain without the control is not a result.
TS7  Sparse-space salience SHALL be a share of the sparsevec INNER PRODUCT (operator 2026-10-04: "i don't want to use bm25, I want to use
     sparsevec inner product"): a term is a set of vocabulary columns (one for a word; a BPE vocabulary spells a longer word or a phrase with
     several pieces, '##' marking continuations), and its share is the sum over those columns, because an inner product is a sum over
     dimensions. The command reads the stored cosine-view sparsevec, finds neighbours with the HNSW cosine index, and never ranks by BM25.
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from entity_derive import g2_2x2


# ------------------------------------------------------------------------------------------------ sparse
def edge_share_of_term(X: sp.csr_matrix, nbrs: np.ndarray, col: int) -> dict:
    """TS1. Require: X rows L2-normalised (cosine view); nbrs (N, k) neighbour rows per chunk (self allowed, skipped).
    Guarantee: {chunks, edges, shared, shares}: chunks holding the term, lexical edges leaving them, those whose other end also holds
    it, and the term's share of cos(i, j) for each shared edge."""
    Xc = X.tocsc()
    lo, hi = Xc.indptr[col], Xc.indptr[col + 1]
    rows = Xc.indices[lo:hi]
    val = dict(zip(rows.tolist(), Xc.data[lo:hi].tolist()))
    edges, shares = 0, []
    for i in rows:
        for k in nbrs[int(i)]:
            if int(k) == int(i):
                continue
            edges += 1
            if int(k) in val:
                c = float(X[int(i)].multiply(X[int(k)]).sum())
                if c > 0:
                    shares.append(val[int(i)] * val[int(k)] / c)
    return {"chunks": len(rows), "edges": edges, "shared": len(shares), "shares": np.array(shares)}


# ----------------------------------------------------------------------------------------------- coarse
def keyness_by_community(P: sp.csr_matrix, lab: np.ndarray, min_df: int = 20) -> dict:
    """TS2. Require: P (chunks x terms) 0/1, lab the community of each chunk. Guarantee: arrays over terms: df, ok (>= min_df), best
    (community with the largest over-represented G2), g2, share (of the term's chunks in that community), spread (entropy over
    communities, divided by log2 of the community count: 0 = one community, 1 = even)."""
    N, E = P.shape
    K = int(lab.max()) + 1
    C = sp.csr_matrix((np.ones(N), (np.arange(N), lab)), shape=(N, K))
    EC = (P.T @ C).toarray()
    size = np.asarray(C.sum(0)).ravel()
    df = np.asarray(P.sum(0)).ravel()
    ok = df >= min_df
    best, g2, share, spread = np.zeros(E, int), np.zeros(E), np.zeros(E), np.zeros(E)
    for e in np.flatnonzero(ok):
        a = EC[e]
        g = g2_2x2(a, df[e] - a, size - a, N - size - (df[e] - a))
        g = np.where(a / np.maximum(size, 1) > df[e] / N, g, 0.0)
        j = int(g.argmax())
        best[e], g2[e], share[e] = j, g[j], a[j] / df[e]
        p = a[a > 0] / df[e]
        spread[e] = float(-(p * np.log2(p)).sum() / np.log2(K)) if K > 1 else 0.0
    return {"df": df, "ok": ok, "best": best, "g2": g2, "share": share, "spread": spread}


def community_terms(P: sp.csr_matrix, lab: np.ndarray, vocab: list[str], n_terms: np.ndarray | int = 8, min_df: int = 3) -> dict[int, list[tuple[str, float]]]:
    """Spec: plan task T163 (operator 2026-10-07: "the dunning terms should just be next to community labels"). TS2 turned the other way round.
    Require: P (sections x terms) 0/1, lab the community per section, vocab the term text per column, n_terms an int or one count per community.
    Guarantee: {community: [(term, G2)]} best first: the terms OVER-represented in the community against every other section (Dunning G2 of the
    2x2 table), held by at least `min_df` of its sections, and never a term that contains, or sits inside, a better-ranked term of the same list
    (so 'retrieval' and 'retrieval augmented' do not both take a slot)."""
    N, K = P.shape[0], int(lab.max()) + 1
    C = sp.csr_matrix((np.ones(N), (np.arange(N), lab)), shape=(N, K))
    EC = (P.T @ C).tocsc()
    size = np.asarray(C.sum(0)).ravel()
    df = np.asarray(P.sum(0)).ravel()
    out: dict[int, list[tuple[str, float]]] = {}
    for c in range(K):
        want = int(n_terms[c]) if hasattr(n_terms, "__len__") else int(n_terms)
        lo, hi = EC.indptr[c], EC.indptr[c + 1]
        e, a = EC.indices[lo:hi], EC.data[lo:hi]
        keep = (a >= min_df) & (a / size[c] > df[e] / N)
        e, a = e[keep], a[keep]
        g = g2_2x2(a, size[c] - a, df[e] - a, N - size[c] - (df[e] - a))
        picked: list[tuple[str, float]] = []
        for j in np.argsort(-g, kind="stable"):
            t = vocab[int(e[j])]
            if any(t in p or p in t for p, _ in picked):
                continue
            picked.append((t, float(g[j])))
            if len(picked) == want:
                break
        out[c] = picked
    return out


def bold_terms(text: str, terms: list[str]) -> tuple[str, list[str]]:
    """Spec: plan task T164 (operator 2026-10-07: "bold any dunning terms that survive into the llm summary phrase"). Guarantee: (text with every whole-word,
    case-insensitive occurrence of a term wrapped in **...** exactly as it appears in the text, the terms that occurred, in the order given). One pass over
    an alternation, longest term first, so 'reinforcement learning' is bolded whole and never nested inside 'learning'; an empty list changes nothing."""
    used = [t for t in terms if re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", text, re.I)]
    if not used:
        return text, []
    pat = re.compile(r"(?<!\w)(" + "|".join(re.escape(t) for t in sorted(used, key=len, reverse=True)) + r")(?!\w)", re.I)
    return pat.sub(lambda m: "**" + m.group(1) + "**", text), used


def exemplar_term_count(n_exemplars: int) -> int:
    """Spec: plan task T163 (operator design: 3 words for a medoid alone, 5 with one more exemplar, 8 with two). Guarantee: 3, 5 or 8."""
    return {1: 3, 2: 5}.get(n_exemplars, 8)


# ------------------------------------------------------------------------------------------------ dense
def remove_term(text: str, term: str) -> str | None:
    """TS3. Guarantee: text with every whole-word, case-insensitive occurrence of `term` deleted and whitespace collapsed, or None
    when the term does not occur."""
    pat = re.compile(r"\b" + re.escape(term) + r"\b", re.I)
    if not pat.search(text):
        return None
    return re.sub(r"\s+", " ", pat.sub(" ", text)).strip()


def occlusion_deltas(encode, texts: list[str], term: str) -> np.ndarray:
    """TS3. Require: encode(list[str]) -> (n, d) L2-normalised. Guarantee: (m, d) array of e(d) - e(d without term) for the m texts
    that hold the term (m = 0 gives a (0, 0) array)."""
    kept = [(t, remove_term(t, term)) for t in texts]
    kept = [(a, b) for a, b in kept if b is not None]
    if not kept:
        return np.zeros((0, 0))
    E = encode([a for a, _ in kept] + [b for _, b in kept])
    n = len(kept)
    return E[:n] - E[n:]


def _unit(X: np.ndarray) -> np.ndarray:
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


def stability(deltas: dict, seed: int = 0) -> dict:
    """TS4. Require: deltas {term: (m, d)} with m >= 2. Guarantee: {same, diff, n_same, n_diff, per_term}: median cosine between two
    deltas of the same term, median cosine between a delta of one term and a delta of another (the null), the pair counts, and each
    term's own median."""
    rng = np.random.default_rng(seed)
    terms = [t for t, D in deltas.items() if len(D) >= 2]
    allt = np.concatenate([np.full(len(deltas[t]), i) for i, t in enumerate(terms)])
    U = _unit(np.concatenate([deltas[t] for t in terms]))
    same, diff, per = [], [], {}
    for i, t in enumerate(terms):
        ix = np.flatnonzero(allt == i)
        S = U[ix] @ U[ix].T
        s = S[np.triu_indices(len(ix), 1)]
        per[t] = float(np.median(s))
        same.append(s)
        others = np.flatnonzero(allt != i)
        pick = rng.choice(others, size=min(len(s), len(others)), replace=False)
        diff.append(np.einsum("ij,ij->i", U[rng.choice(ix, size=len(pick))], U[pick]))
    same, diff = np.concatenate(same), np.concatenate(diff)
    return {"same": float(np.median(same)), "diff": float(np.median(diff)), "n_same": len(same), "n_diff": len(diff), "per_term": per}


def token_share_vector(V: np.ndarray, sel: list[int]) -> np.ndarray:
    """TS5. Require: V (n_tokens, d) token vectors before pooling; sel the term's token rows. Guarantee: sum of V[sel] over n_tokens."""
    return V[sel].sum(0) / len(V) if sel else np.zeros(V.shape[1])


def term_direction(D: np.ndarray) -> np.ndarray:
    """TS6. Guarantee: e*(t), the mean of the term's deltas."""
    return D.mean(0)


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", a, b) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-9)


def subtraction_gain(Ed: np.ndarray, Ew: np.ndarray, right: np.ndarray, wrong: np.ndarray) -> dict:
    """TS6. Require: Ed e(chunk), Ew e(chunk without t), right the e*(t) of each chunk's term, wrong a different term's e*.
    Guarantee: median cosine to Ew before and after subtracting `right` and `wrong`, and the share of pairs each improves."""
    base = _cos(Ed, Ew)
    r, w = _cos(Ed - right, Ew) - base, _cos(Ed - wrong, Ew) - base
    return {"base": float(np.median(base)), "right_gain": float(np.median(r)), "wrong_gain": float(np.median(w)),
            "right_improved": float((r > 0).mean()), "wrong_improved": float((w > 0).mean()), "n": len(base)}


# --------------------------------------------------------------------------------------------- command line
def parse_sparsevec(text: str) -> dict[int, float]:
    """TS7. Guarantee: {0-based column: weight} from pgvector's '{col:weight,...}/dim' text (the columns there are 1-based)."""
    body = text.split("/")[0].strip("{}")
    return {int(a) - 1: float(b) for a, b in (kv.split(":") for kv in body.split(",") if kv)}


def community_profile(counts: np.ndarray, sizes: np.ndarray, top: int = 5) -> dict:
    """TS7. Require: counts[c] = the term's chunks in community c, sizes[c] = chunks in community c. Guarantee: {n, spread, top}:
    spread is the entropy of the term over the communities divided by log2 of their number (0 = one community, 1 = even); top lists
    (community, term chunks there, share of the term's chunks, share of that community's chunks, G2 over-representation), by G2."""
    N, n = int(sizes.sum()), int(counts.sum())
    if n == 0:
        return {"n": 0, "spread": 0.0, "top": []}
    p = counts[counts > 0] / n
    spread = float(-(p * np.log2(p)).sum() / np.log2(len(sizes))) if len(sizes) > 1 else 0.0
    g = g2_2x2(counts, n - counts, sizes - counts, N - sizes - (n - counts))
    g = np.where(counts / np.maximum(sizes, 1) > n / N, g, 0.0)
    order = [c for c in np.argsort(-g)[:top] if counts[c] > 0 and g[c] > 0]
    return {"n": n, "spread": spread, "top": [(int(c), int(counts[c]), float(counts[c] / n), float(counts[c] / sizes[c]), float(g[c])) for c in order]}


def term_columns(vocab: dict[str, int], term: str) -> list[int] | None:
    """TS7. Require: vocab {piece: 0-based column}; a BPE vocabulary marks continuation pieces with '##'. Guarantee: the columns that spell
    `term` (lowercased, split on whitespace): a word that is one piece gives that column; any other word is split greedily, longest
    leading piece first and '##' continuations after it, so a whole word or phrase becomes the set of its pieces' columns. None when a
    word cannot be spelled from the vocabulary."""
    cols: list[int] = []
    for w in term.lower().split():
        if w in vocab:
            cols.append(vocab[w])
            continue
        i, spelled = 0, []
        while i < len(w):
            for j in range(len(w), i, -1):
                piece = w[i:j] if i == 0 else "##" + w[i:j]
                if piece in vocab:
                    spelled.append(vocab[piece])
                    i = j
                    break
            else:
                return None
        cols.extend(spelled)
    return sorted(set(cols)) or None


def self_share(vec: dict[int, float], cols: list[int]) -> float:
    """TS7. Guarantee: the share of a vector's own inner product <v, v> that the columns carry: sum of v_c^2 over them / sum of all v_c^2
    (the L2-normalised cosine view has <v, v> = 1, so this is the term's squared weight). The shares of all columns sum to 1."""
    tot = sum(w * w for w in vec.values())
    return sum(vec.get(c, 0.0) ** 2 for c in cols) / tot if tot > 0 else 0.0


def pair_share(a: dict[int, float], b: dict[int, float], cols: list[int]) -> tuple[float, float]:
    """TS1, TS7. Guarantee: (share, ip): the inner product <a, b> over every shared column, and the part of it the `cols` carry. Because an
    inner product is a sum over dimensions, a whole word or phrase spelled by several BPE pieces carries exactly the sum of its pieces'
    shares. share is 0 when ip <= 0."""
    ip = sum(w * b[c] for c, w in a.items() if c in b)
    if ip <= 0:
        return 0.0, ip
    return sum(a.get(c, 0.0) * b.get(c, 0.0) for c in cols) / ip, ip


def profile_term(conn, label: str, term: str, top: int = 5, k: int = 15) -> dict:
    """TS7. Guarantee: what the live Postgres build says about `term`, in sparsevec inner-product terms. The term is spelled as columns of the
    label's vocabulary (term_columns); a chunk holds it when its stored cosine-view sparsevec is nonzero on every column. Per chunk: the
    term's share of the chunk's own inner product (self_share), and over its k nearest sparsevec neighbours (HNSW cosine) the share of the
    chunk-neighbour inner product the term carries (pair_share). Plus the term's spread over the live build's communities. {} when the
    term cannot be spelled from the vocabulary or no chunk holds it."""
    import sparsevec_store as ss
    b = ss.live_build(conn, label)
    assert b is not None, "no live build for %r" % label
    build, dim = b[0], b[2]["bm25"]["dim"]
    vocab = {piece: int(c) - 1 for c, piece in conn.execute("SELECT col, piece FROM lex_vocab WHERE label = %s", (label,)).fetchall()}
    cols = term_columns(vocab, term)
    if cols is None:
        return {}
    cos_t = ss._table(label, "cos")
    where = " AND ".join("vec::text ~ %s" for _ in cols)
    rows = conn.execute(f"SELECT ord, vec::text FROM {cos_t} WHERE {where}", tuple(r"[{,]%d:" % (c + 1) for c in cols)).fetchall()
    if not rows:
        return {}
    vecs = {o: parse_sparsevec(txt) for o, txt in rows}
    ranked = sorted(vecs, key=lambda o: -self_share(vecs[o], cols))
    ords = list(vecs)
    cids = dict(conn.execute("SELECT ord, cid FROM lex_assign WHERE build_id = %s AND ord = ANY(%s)", (build, ords)).fetchall())
    sizes = np.array([r[1] for r in conn.execute("SELECT cid, count(*) FROM lex_assign WHERE build_id = %s GROUP BY cid ORDER BY cid", (build,)).fetchall()])
    counts = np.zeros(len(sizes))
    for o in ords:
        if o in cids:
            counts[cids[o]] += 1
    meta = {r[0]: (r[1], r[2]) for r in conn.execute("SELECT ord, doc_id, text FROM lex_chunk_meta WHERE label = %s AND ord = ANY(%s)", (label, ranked[:top])).fetchall()}
    first = term.lower().split()[0]
    best = []
    pooled, shared_n, total_n = [], 0, 0
    for o in ranked[:top]:
        lit = "{" + ",".join("%d:%r" % (c + 1, w) for c, w in sorted(vecs[o].items())) + "}/%d" % dim
        nb = conn.execute(f"SELECT ord, vec::text FROM {cos_t} WHERE ord <> %s ORDER BY vec <=> %s::sparsevec({dim}) LIMIT %s", (o, lit, k)).fetchall()
        shares = []
        for _, txt in nb:
            nv = parse_sparsevec(txt)
            total_n += 1
            if all(nv.get(c, 0.0) > 0 for c in cols):
                shared_n += 1
                shares.append(pair_share(vecs[o], nv, cols)[0])
        pooled += shares
        doc, text = meta.get(o, ("?", ""))
        i = text.lower().find(first)
        best.append({"ord": o, "doc": doc, "self_share": self_share(vecs[o], cols), "neighbours_with_term": len(shares), "neighbours": len(nb),
                     "median_pair_share": float(np.median(shares)) if shares else 0.0,
                     "snippet": " ".join(text[max(i - 70, 0): i + 90].split()) if i >= 0 else " ".join(text[:160].split())})
    papers = len({r[0] for r in conn.execute("SELECT doc_id FROM lex_chunk_meta WHERE label = %s AND ord = ANY(%s)", (label, ords)).fetchall()})
    return {"term": term, "columns": [{"col": c + 1, "piece": next((p for p, v in vocab.items() if v == c), "?")} for c in cols],
            "chunks": len(ords), "papers": papers, "best": best, "communities": community_profile(counts, sizes, top),
            "pooled_pair_share": float(np.median(pooled)) if pooled else 0.0, "neighbours_sharing": shared_n, "neighbours_total": total_n}


def dense_profile(conn, label: str, term: str, sample: int = 30, seed: int = 0) -> dict:
    """TS3, TS4. Guarantee: the term's effect on the MiniLM embedding of up to `sample` randomly drawn chunks that hold it in their first 600
    characters (the 256-token window): median |e(d) - e(d without term)| and the median cosine between two of those deltas. {} when too few."""
    from arxiv_community_map import encode_minilm
    pat = re.compile(r"\b" + re.escape(term) + r"\b", re.I)
    rows = conn.execute("SELECT text FROM lex_chunk_meta WHERE label = %s AND NOT is_reference AND NOT is_junk AND text ~* %s", (label, r"\m" + re.escape(term) + r"\M")).fetchall()
    texts = [r[0] for r in rows if (m := pat.search(r[0])) and m.start() < 600]
    if len(texts) < 4:
        return {}
    rng = np.random.default_rng(seed)
    pick = [texts[i] for i in rng.permutation(len(texts))[:sample]]
    D = occlusion_deltas(encode_minilm, pick, term)
    U = _unit(D)
    S = U @ U.T
    return {"n": len(D), "median_delta": float(np.median(np.linalg.norm(D, axis=1))), "median_pair_cosine": float(np.median(S[np.triu_indices(len(D), 1)]))}


def main(argv: list[str] | None = None) -> None:
    import argparse
    import io
    import sparsevec_store as ss
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="backslashreplace", line_buffering=True)
    ap = argparse.ArgumentParser(description="What a term does in the sparsevec space of the live Postgres build: its share of the inner product between a chunk and "
                                 "its sparsevec neighbours, its spread over communities, and (--dense) its effect on the embedding.")
    ap.add_argument("terms", nargs="+")
    ap.add_argument("--label", default="arxiv_sect")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--k", type=int, default=15, help="sparsevec neighbours per chunk")
    ap.add_argument("--dense", action="store_true", help="also embed with MiniLM and measure the term's effect (needs the GPU, ~10 s)")
    a = ap.parse_args(argv)
    conn = ss.connect()
    for term in a.terms:
        p = profile_term(conn, a.label, term, a.top, a.k)
        print("=" * 100)
        if not p:
            print("%s: cannot be spelled from the vocabulary, or no chunk holds every one of its columns" % term)
            continue
        print("%s   columns %s   %d chunks in %d papers" % (term, ", ".join("%d=%s" % (c["col"], c["piece"]) for c in p["columns"]), p["chunks"], p["papers"]))
        print("  across the %d nearest sparsevec neighbours of the %d most salient chunks: %d hold the term; where they do, it carries a median %.1f%% of the inner product" % (
            p["neighbours_total"], len(p["best"]), p["neighbours_sharing"], 100 * p["pooled_pair_share"]))
        c = p["communities"]
        print("  spread over communities: %.2f (0 = one community, 1 = even), top communities:" % c["spread"])
        for cid, n, share, density, g in c["top"]:
            print("    community %-4d %5d chunks = %4.0f%% of the term, %4.1f%% of that community, G2 %.0f" % (cid, n, 100 * share, 100 * density, g))
        print("  most salient chunks (share of the chunk's own inner product | neighbours holding the term, median share of the inner product with them):")
        for b in p["best"]:
            print("    %5.1f%% | %2d/%2d, %5.1f%%   %-22s ...%s..." % (100 * b["self_share"], b["neighbours_with_term"], b["neighbours"],
                                                                   100 * b["median_pair_share"], b["doc"], b["snippet"]))
        if a.dense:
            d = dense_profile(conn, a.label, term)
            print("  embedding effect (MiniLM, n=%d chunks): median |delta| %.4f (deleting a random word: about 0.09), median cosine between two deltas %.3f (different terms: about 0.00)" % (
                d["n"], d["median_delta"], d["median_pair_cosine"]) if d else "  embedding effect: too few chunks hold the term in the first 600 characters")


if __name__ == "__main__":
    main()
