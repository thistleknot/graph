"""Build the current graph sections into a separate section-keyed BM25 sparsevec index.

| Stage | Mechanism | Tag | Ref |
|---|---|---|---|
| SOURCE | Read all rows from the active `sect_node` build and chunk each section independently. | [mandatory] | R1 |
| SAMPLE | Persist a deterministic 80k materialized sample in the BM25 database. | [mandatory] | R2 |
| TOKENIZE | Train BPE from the sample and spell every population term. | [mandatory] | R3 |
| SCORE | Estimate sample DF at population scale; build BM25 weighted BPE sparse vectors. | [mandatory] | R4 |
| STORE | Persist tokenizer, IDF, metadata, sample provenance, and narrow sparse vectors to PostgreSQL. | [mandatory] | R5 |
| PUBLISH | Validate all generation rows, then atomically make it live. | [mandatory] | R6 |
| MEASURE | Verify exact-query parity and bounded recall diagnostics when requested. | [opt] | R7 |

R1 Each chunk preserves its live graph `(doc_id, section_idx)` key; sections never merge.
R2 The sample materialized view is deterministic for a source corpus and seed; smaller corpora use all rows.
R3 BPE-only eligibility excludes sample df=1 and overlong non-dictionary words beyond median + 2*1.4826*MAD letter length; all terms remain encoded for BM25.
R4 Sample df is rounded after scaling by population/sample size and clamped to [1, N]; unseen sample terms use df=1.
R5 The graph DB supplies source sections; `ARXIV_BM25_DSN` owns BM25 artifacts. The tokenizer and source graph build are frozen in `lex_build`.
R6 A failed/incomplete build SHALL remain non-live; a completed build has exactly one sparse vector per metadata row, including empty encodings.
R7 Defaults are sample=80k, BPE=10000, k1=1.5, b=0.75, exact sparsevec scan, and top 144 candidates before adjacency reranking.

CLOSED: the prior `.tmp/arxiv_section_chunks.pkl` and existing `arxiv_chunks` database rows are rejected as live sources because their keys do not cover the active graph build. `tsvector` is not a fallback.

Preconditions: graph and BM25 PostgreSQL endpoints are reachable; an active graph section build exists. Failure modes: missing source/build, incomplete sample, unspellable term, or database failure aborts the build.

Run: `python -u src\\ingest_arxiv_sparsevec.py --from-live-graph --label arxiv_sect_bpe --bpe 10000 --no-battery`; default mode reads the legacy diagnostic cache.
"""
from __future__ import annotations

import collections
import io
import math
import os
import pickle
import re
import sys
import time

from wordfreq import zipf_frequency

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import scipy.sparse as ss_sparse

import sparsevec_store as ss
from diag_dense_arms import pick_sentence
from domain_corpora import with_junk_flag
from stoplist import tokenize

CACHE = ".tmp/arxiv_section_chunks.pkl"
LABEL = "arxiv_sect"
K, N_Q, SEED = 50, 200, 0
K1, B = 1.5, 0.75                   # salient_grams.bm25_matrix's defaults, passed explicitly so a build records them
DENSE_DIM = 384                     # all-MiniLM-L6-v2


def log(msg: str) -> None:
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def saturated(BM, df, N):
    """R6. Guarantee: (sat, idf) with sat[i, t] * idf[t] == BM[i, t]; sat is CSR."""
    idf = np.log(1 + (N - df + 0.5) / (df + 0.5))
    sat = BM.tocsr().copy()
    sat.data = sat.data / idf[sat.indices]
    return sat, idf


def keep_heaviest(BM, cap: int = ss.SPARSEVEC_MAX_NNZ):
    """V2. Guarantee: boolean mask over BM.data keeping each row's `cap` heaviest by BM weight."""
    mask = np.ones(BM.nnz, dtype=bool)
    for i in np.where(np.diff(BM.indptr) > cap)[0]:
        s, e = BM.indptr[i], BM.indptr[i + 1]
        mask[s + np.argsort(-BM.data[s:e], kind="stable")[cap:]] = False
    return mask


def full_views(BM, df, N, cap: int = ss.SPARSEVEC_MAX_NNZ):
    """V2, V7, R7. Guarantee: (sat_t, cos_t, idf, whales). sat_t: saturated tf with each row's `cap`
    heaviest terms by BM weight (the inner-product view, idf on the query side). cos_t: the symmetric
    doc-doc view -- those same kept terms as idf*sat, terms in more than half the documents dropped
    (`whales`, columns), rows L2-normalised. doc_row reproduces one row of each from frozen statistics."""
    sat, idf = saturated(BM, df, N)
    mask = keep_heaviest(BM, cap)
    sat_t = sat.copy()
    sat_t.data = sat.data * mask
    sat_t.eliminate_zeros()
    whales = np.where(df * 2 > N)[0]
    cos_t = BM.tocsr().copy()
    cos_t.data = cos_t.data * mask * ~np.isin(cos_t.indices, whales)
    cos_t.eliminate_zeros()
    norms = np.sqrt(np.asarray(cos_t.multiply(cos_t).sum(1)).ravel())
    norms[norms == 0] = 1.0
    return sat_t, (ss_sparse.diags(1.0 / norms) @ cos_t).tocsr(), idf, whales


def doc_row(tokens: list[str], col: dict, idf, avgdl: float, whales=(), cap: int = ss.SPARSEVEC_MAX_NNZ):
    """Guarantee: ({col0: saturated tf}, {col0: L2-normalised idf*sat}) for ONE document under FROZEN
    statistics (vocabulary `col`, idf, avgdl, whales), equal to the row full_views gives it in the
    build that fixed them. Terms outside the vocabulary are dropped; `tokens` is every token of the
    document, because document length counts them all."""
    cnt = collections.Counter(t for t in tokens if t in col)
    if not cnt:
        return {}, {}
    norm_len = 1 - B + B * len(tokens) / avgdl
    sat = {col[t]: c * (K1 + 1) / (c + K1 * norm_len) for t, c in cnt.items()}
    kept = {j: sat[j] for j in sorted(sat, key=lambda j: (-sat[j] * idf[j], j))[:cap]}      # ties go to the lower column, as keep_heaviest's stable sort does
    bm = {j: s * idf[j] for j, s in kept.items() if j not in set(whales)}
    n = math.sqrt(sum(w * w for w in bm.values()))
    return kept, ({j: w / n for j, w in bm.items()} if n > 0 else {})


def gold(recs: list[dict]):
    """Guarantee: ([(title, relevant_ords)], [(sentence, relevant_ords)]), seeded, n=N_Q each."""
    by_doc = collections.defaultdict(set)
    first = {}
    for i, r in enumerate(recs):
        by_doc[r["doc_id"]].add(i)
        first.setdefault(r["doc_id"], r["section_title"])
    rng = np.random.default_rng(SEED)
    docs = sorted(d for d, t in first.items() if not t.startswith("arxiv/") and len(t.split()) >= 3)
    titles = [(first[d], by_doc[d]) for d in rng.permutation(docs)[:N_Q]]
    body = []
    for i in rng.permutation(len(recs)):
        if recs[i]["is_reference"]:
            continue
        got = pick_sentence(recs[i]["text"], rng)
        if got:
            body.append((got[0], by_doc[recs[i]["doc_id"]]))
        if len(body) == N_Q:
            break
    return titles, body


def bpe_docs(token_docs: list[list[str]], merges: int, return_tokenizer: bool = False,
             training_token_docs: list[list[str]] | None = None):
    """Train BPE on eligible terms, then encode every document and query term.

    Require:
        `token_docs` contains the complete word streams used for BM25.
    Guarantee:
        Returns the same document streams spelled as BPE pieces and a query spelling function, optionally followed by the tokenizer.
        Training excludes df=1 terms and overlong non-dictionary terms; encoding and BM25 retain every source term.
    Maintain:
        The BPE training gate never removes a term from document/query streams or their later BM25 counts.
    Assert:
        Every corpus word is spellable without `[UNK]`.
    """
    import hnsw_communities as hc
    training_token_docs = token_docs if training_token_docs is None else training_token_docs
    df = collections.Counter(w for d in training_token_docs for w in set(d))
    n = sum(1 for d in training_token_docs if d)
    lengths = [len(re.sub(r"[^a-z]", "", w.lower())) for w in df]
    letter_lengths = [length for length in lengths if length > 0]
    median = float(np.median(letter_lengths)) if letter_lengths else 0.0
    mad = float(np.median(np.abs(np.asarray(letter_lengths) - median))) if letter_lengths else 0.0
    max_length = median + 2 * 1.4826 * mad

    def trains_merges(word: str, count: int) -> bool:
        if count < 2 or count > 0.5 * n:
            return False
        letters = len(re.sub(r"[^a-z]", "", word.lower()))
        return letters <= max_length or zipf_frequency(word, "en") > 0

    # Compute eligibility only for merge training. The complete word list below still
    # gets encoded, so tokenizer cleanup cannot silently become a BM25 term filter.
    survivors = sorted(w for w, c in df.items() if trains_merges(w, c))
    norm = hc._NORM.normalize_str
    have = {ch for w in survivors for ch in norm(w)}
    missing = sorted({ch for w in df for ch in norm(w) if not ch.isspace()} - have)
    tok = hc.train_bpe(survivors + missing, vocab=merges)  # isolated characters extend the alphabet without creating outlier merges
    words = sorted({word for doc in token_docs for word in doc})
    spelled = {w: e.tokens for w, e in zip(words, tok.encode_batch(words))}
    assert not any("[UNK]" in t for t in spelled.values()), "a word was spelled with [UNK]: the alphabet does not cover the corpus"

    def to_pieces(ws: list[str]) -> list[str]:
        return [p for w in ws for p in (spelled[w] if w in spelled else tok.encode(w).tokens)]
    result = ([to_pieces(d) for d in token_docs], to_pieces)
    return (*result, tok) if return_tokenizer else result


def sampled_bm25_matrix(token_docs: list[list[str]], sample_indices: list[int],
                        k1: float = K1, b: float = B):
    """Build population BM25 vectors using scaled document frequencies from the persisted sample.

    Require:
        `sample_indices` are unique valid rows selected by the durable sample view.
    Guarantee:
        Returns `(BM, TF, terms, df, sample_df, avgdl)`; unseen terms use df=1 and sample df is scaled to population size.
    Maintain:
        Sampling changes only document-frequency estimates, not the source token streams or their term frequencies.
    Assert:
        Every estimated document frequency is in [1, number of source documents].

    Args:
        token_docs: Population-wide token streams, including empty junk rows.
        sample_indices: Durable sample materialized-view ordinals.
        k1: BM25 term-frequency saturation constant.
        b: BM25 document-length normalization constant.

    Returns:
        BM25 and term-frequency CSR matrices, vocabulary, scaled DF, sampled DF, and average length.

    Raises:
        ValueError: If the population or sample is empty or sample indexes are invalid.
    """
    if not token_docs or not sample_indices:
        raise ValueError("BM25 source and sample must each contain at least one row")
    if len(set(sample_indices)) != len(sample_indices) or min(sample_indices) < 0 or max(sample_indices) >= len(token_docs):
        raise ValueError("sample indices must be unique source row indexes")
    vocab: dict[str, int] = {}
    row_ids, col_ids, values = [], [], []
    for row, tokens in enumerate(token_docs):
        counts = collections.Counter(tokens)
        for term, count in counts.items():
            col = vocab.setdefault(term, len(vocab))
            row_ids.append(row)
            col_ids.append(col)
            values.append(count)
    if not vocab:
        raise ValueError("empty vocabulary after tokenization")
    terms = np.asarray(sorted(vocab, key=vocab.get))
    tf = ss_sparse.csr_matrix((values, (row_ids, col_ids)), shape=(len(token_docs), len(terms)))
    sample_df = np.asarray((tf[sample_indices] > 0).sum(0)).ravel().astype(np.int64)
    population_n = len(token_docs)
    sample_n = len(sample_indices)
    estimated_df = np.rint(sample_df * population_n / sample_n).astype(np.int64)
    estimated_df = np.clip(estimated_df, 1, population_n)
    idf = np.log(1.0 + (population_n - estimated_df + 0.5) / (estimated_df + 0.5))
    doc_lengths = np.asarray([len(tokens) for tokens in token_docs], dtype=np.float64)
    avgdl = float(doc_lengths.mean()) or 1.0
    coo = tf.tocoo()
    weights = (idf[coo.col] * coo.data * (k1 + 1.0) /
               (coo.data + k1 * (1.0 - b + b * doc_lengths[coo.row] / avgdl)))
    bm = ss_sparse.csr_matrix((weights, (coo.row, coo.col)), shape=tf.shape)
    assert np.all((estimated_df >= 1) & (estimated_df <= population_n))
    return bm, tf, terms, estimated_df, sample_df, avgdl


def live_graph_chunks(graph_conn, graph_tag: str = "xpa", max_sections: int | None = None) -> tuple[list[dict], dict, int]:
    """Read and chunk every section from the active graph build without crossing section boundaries.

    Require:
        The graph database contains a live section build for `graph_tag`.
    Guarantee:
        Returns stable-key BM25 chunk records, the chunk-fit parameters, and source build ID. `max_sections` bounds synthetic/live probes.
    Maintain:
        Every emitted chunk keeps the exact graph `(doc_id, section_idx)` key; each source section is chunked independently.
    Assert:
        No business key is duplicated.

    Args:
        graph_conn: Connection to the active graph database.
        graph_tag: Live graph build tag.
        max_sections: Optional positive section limit for a bounded probe.

    Returns:
        Chunk metadata records, chunk-fit parameters, and source build ID.

    Raises:
        RuntimeError: If no active graph build or section rows exist.
        ValueError: If `max_sections` is not positive.
    """
    import section_store as store
    from domain_corpora import chunk_arxiv_doc, fit_arxiv, with_junk_flag

    live = store.live_build(graph_conn, graph_tag)
    if live is None:
        raise RuntimeError("no live section graph build for %r" % graph_tag)
    build_id, _ = live
    query = "SELECT doc_id, section_idx, section_title, text FROM sect_node WHERE build_id = %s ORDER BY ord"
    if max_sections is not None:
        if max_sections < 1:
            raise ValueError("max_sections must be positive")
        query += " LIMIT %s"
        sections = graph_conn.execute(query, (build_id, max_sections)).fetchall()
    else:
        sections = graph_conn.execute(query, (build_id,)).fetchall()
    if not sections:
        raise RuntimeError("active section graph build %s contains no sections" % build_id)
    fit = fit_arxiv([row[3] for row in sections])
    splitter = None
    from domain_corpora import make_splitter
    splitter = make_splitter(fit)
    records = []
    for doc_id, section_idx, section_title, text in sections:
        source_text = text
        if section_title and not re.match(r"^\s*#{1,6}\s+", text):
            source_text = "## " + section_title + "\n\n" + text
        chunks, _ = chunk_arxiv_doc(doc_id, source_text, fit, splitter)
        for chunk_idx, (chunk_text, _, _, _, is_reference) in enumerate(chunks):
            records.append({"doc_id": doc_id, "version": 1, "section_idx": section_idx,
                            "chunk_idx": chunk_idx, "section_title": section_title or "",
                            "is_reference": is_reference, "is_junk": False, "text": chunk_text})
    records = with_junk_flag(records)
    keys = [(row["doc_id"], row["section_idx"], row["chunk_idx"]) for row in records]
    assert len(keys) == len(set(keys)), "duplicate section chunk key from active graph source"
    return records, fit, build_id


def refresh_sample_view(conn, label: str, sample_size: int, seed: str, build_id: int) -> tuple[str, list[int]]:
    """Persist a deterministic materialized sample and return its name and source ords.

    Args:
        conn: BM25 database connection with current chunk metadata loaded.
        label: Safe lexicon/index label.
        sample_size: Maximum number of metadata rows in the sample.
        seed: Stable salt for deterministic sample ordering.

    Returns:
        Materialized view name and its chunk ordinals in stable order.

    Raises:
        ValueError: If the BM25 metadata population is empty.
    """
    view = "lex_sample_" + "".join(ch if ch.isalnum() else "_" for ch in label.lower())
    seed_sql = seed.replace("'", "''")
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM lex_chunk_meta WHERE label = %s AND build_id = %s", (label, int(build_id)))
        population = int(cur.fetchone()[0])
        if population == 0:
            raise ValueError("cannot sample an empty BM25 metadata population")
        cur.execute("DROP MATERIALIZED VIEW IF EXISTS " + view)
        cur.execute(
            "CREATE MATERIALIZED VIEW " + view + " AS "
            "SELECT build_id, ord, doc_id, section_idx, chunk_idx, text FROM ("
            "SELECT build_id, ord, doc_id, section_idx, chunk_idx, text, "
            "row_number() OVER (PARTITION BY build_id ORDER BY md5(doc_id || ':' || section_idx::text || ':' || chunk_idx::text || ':' || '" + seed_sql + "'), "
            "doc_id, section_idx, chunk_idx) AS sample_rank FROM lex_chunk_meta WHERE label = '" + label.replace("'", "''") + "') ranked "
            "WHERE sample_rank <= " + str(min(int(sample_size), population))
        )
        cur.execute("CREATE UNIQUE INDEX " + view + "_build_ord ON " + view + " (build_id, ord)")
        cur.execute("SELECT ord FROM " + view + " WHERE build_id = %s ORDER BY ord", (int(build_id),))
        ords = [int(row[0]) for row in cur.fetchall()]
    return view, ords


def query_terms(text: str, col: dict, idf: np.ndarray, tok=tokenize):
    """R2, R6. Guarantee: (cols, weights = idf * qtf) over the terms the vocabulary holds; `tok` spells the query as the index was spelled."""
    qtf = collections.Counter(t for t in tok(text) if t in col)
    cols = sorted(col[t] for t in qtf)
    inv = {col[t]: n for t, n in qtf.items()}
    return cols, [float(idf[c] * inv[c]) for c in cols]


def main() -> None:
    global LABEL
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    if "--label" in sys.argv:                                      # a side label never touches the live one (A6)
        LABEL = sys.argv[sys.argv.index("--label") + 1]
    import config
    from domain_corpora import with_junk_flag

    live_mode = "--from-live-graph" in sys.argv
    bpe = int(sys.argv[sys.argv.index("--bpe") + 1]) if "--bpe" in sys.argv else (10000 if live_mode else None)
    assert bpe is None or LABEL != "arxiv_sect", "a BPE build goes to a side label, never over the live arxiv_sect"
    graph_conn = None
    graph_build_id = None
    fit = None
    if live_mode:
        graph_conn = ss.connect(config.DSN)
        recs, fit, graph_build_id = live_graph_chunks(graph_conn)
    else:
        cached = pickle.load(open(CACHE, "rb"))
        recs = with_junk_flag(cached["records"])
        fit = cached.get("fit")
    log("chunks %d (reference-flagged %d, junk-flagged %d)"
        % (len(recs), sum(r["is_reference"] for r in recs), sum(r["is_junk"] for r in recs)))

    t0 = time.time()
    token_docs = [[] if r["is_junk"] else tokenize(r["text"]) for r in recs]       # A4: a junk chunk is an empty document
    tok_fn = tokenize
    tokenizer_json = None
    bm25_conn = ss.connect(config.ARXIV_BM25_DSN)
    ss.ensure_generation_schema(bm25_conn, LABEL)
    build = ss.open_generation(bm25_conn, LABEL, len(recs), {
        "source_graph_build": graph_build_id,
        "tokenizer": {"kind": "pending", "merges": bpe},
    })
    log("generation %d opened as non-live" % build)
    log("meta written: %d rows" % ss.write_generation_meta(bm25_conn, LABEL, build, recs))
    sample_view, sample_ords = refresh_sample_view(
        bm25_conn, LABEL, config.ARXIV_BM25_SAMPLE_SIZE, config.ARXIV_BM25_SAMPLE_SEED, build)
    sample_indices = sample_ords  # metadata ordinals are the deterministic source row offsets
    sample_token_docs = [token_docs[i] for i in sample_indices]
    log("sample persisted: %s rows=%d population=%d seed=%s" %
        (sample_view, len(sample_indices), len(recs), config.ARXIV_BM25_SAMPLE_SEED))
    if bpe:
        token_docs, to_pieces, bpe_tokenizer = bpe_docs(
            token_docs, bpe, return_tokenizer=True, training_token_docs=sample_token_docs)
        tokenizer_json = bpe_tokenizer.to_str()
        sample_token_docs = [token_docs[i] for i in sample_indices]
        tok_fn = lambda text: to_pieces(tokenize(text))
        log("BPE vocabulary: merge budget %d, sample-trained, all %d chunks encoded" % (bpe, len(token_docs)))
    BM, TF, terms, df, sample_df, avgdl = sampled_bm25_matrix(
        token_docs, sample_indices, K1, B)
    BM = BM.tocsr()
    N, dim = BM.shape
    sat_t, _, idf, _ = full_views(BM, df, N)
    n_cut = int((np.diff(BM.indptr) > ss.SPARSEVEC_MAX_NNZ).sum())
    log("bm25 %.0fs  rows=%d dim=%d nnz=%d  rows over %d nnz trimmed: %d"
        % (time.time() - t0, N, dim, BM.nnz, ss.SPARSEVEC_MAX_NNZ, n_cut))

    conn = bm25_conn
    params = {
        "bm25": {"k1": K1, "b": B, "avgdl": avgdl, "n_docs": N, "dim": dim,
                 "df_sample_size": len(sample_indices), "sample_population": N,
                 "sample_scale": N / len(sample_indices), "sample_view": sample_view,
                 "sample_seed": config.ARXIV_BM25_SAMPLE_SEED,
                 "sample_build_id": build},
        "fit": fit, "source_graph_build": graph_build_id, "dense_dim": DENSE_DIM,
        "tokenizer": ({"kind": "bpe", "merges": bpe, "json": tokenizer_json} if bpe else {"kind": "words"})}
    ss.update_build_params(conn, build, params)
    t0 = time.time()
    ss.write_generation_vocab(conn, build, [str(t) for t in terms], idf)
    log("build-scoped vocab written  %.0fs" % (time.time() - t0))

    t0 = time.time()
    w = ss.write_generation_vectors(conn, LABEL, build, dim, (
        dict(zip(sat_t.indices[sat_t.indptr[i]:sat_t.indptr[i + 1]].tolist(),
                 sat_t.data[sat_t.indptr[i]:sat_t.indptr[i + 1]].tolist()))
        for i in range(N)
    ))
    log("exact vectors written %d rows (empty vectors retained, truncated=%d)  %.0fs"
        % (w["written"], w["truncated"], time.time() - t0))
    if graph_conn is not None:
        import section_store as store
        current_graph = store.live_build(graph_conn, "xpa")
        if current_graph is None or current_graph[0] != graph_build_id:
            raise RuntimeError("source section build changed during BM25 generation; leaving it unpublished")
    ss.publish_generation(conn, LABEL, build, len(recs), dim)
    log("generation %d validated and published" % build)

    if "--no-battery" in sys.argv:
        log("--no-battery: recall measurements skipped")
        conn.close()
        if graph_conn:
            graph_conn.close()
        return
    col = {str(t): j for j, t in enumerate(terms)}
    sat_c = sat_t.tocsc()
    titles, body = gold(recs)
    log("gold: title %d  body %d" % (len(titles), len(body)))
    hdr = "  %-24s %8s %9s"
    for gname, gq in (("title -> own paper", titles), ("body sentence -> own paper", body)):
        qs = [(query_terms(q, col, idf, tok_fn), rel) for q, rel in gq]
        qs = [(c, w_, rel) for (c, w_), rel in qs if c]
        print("\n  [%s]  n=%d (queries with no vocabulary term dropped: %d)" % (gname, len(qs), len(gq) - len(qs)))
        print(hdr % ("arm", "rec@50", "ms/q"))

        def recall(top):
            return float(np.mean([len(set(t) & rel) / len(rel) for t, (_, _, rel) in zip(top, qs)]))
        np_top, np_scores = [], []
        t0 = time.time()
        for cols, wts, _ in qs:
            s = sat_c[:, cols] @ np.asarray(wts)
            o = np.argsort(-s, kind="stable")[:K]
            np_top.append(o.tolist())
            np_scores.append(s[o])
        print(hdr % ("numpy exact", "%.3f" % recall(np_top), "%.0f" % ((time.time() - t0) / len(qs) * 1000)))
        t0 = time.time()
        pg = [ss.query_generation(conn, LABEL, build, dim, c, K, weights=w_) for c, w_, _ in qs]
        ms = (time.time() - t0) / len(qs) * 1000
        print(hdr % ("psql exact scan", "%.3f" % recall([[o for o, _ in r] for r in pg]), "%.0f" % ms))
        same = np.mean([np.allclose(np.sort(a), np.sort([s for _, s in r]), rtol=1e-3, atol=1e-4) for a, r in zip(np_scores, pg)])
        print("  parity: psql exact top-%d SCORE set == numpy on %.1f%% of queries%s"
              % (K, 100 * same, "" if same >= 0.99 else "   <-- PARITY DEFECT, read no HNSW number below"))
    conn.close()
    if graph_conn:
        graph_conn.close()


if __name__ == "__main__":
    main()
