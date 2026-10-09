"""diag_dense_arms.py -- which dense embedding can find a chunk of the arxiv corpus again?

NO GOVERNING SPEC. Basis: operator instruction 2026-10-02 ("for embeddings I would use
model2vec and compress minilm-l6 to 128 dims ... idk if this old school embedding can
handle technical terms in arxiv papers ... I might benefit from some gemma model").
Chunks: domain_corpora.chunk_arxiv, cached by .tmp/run_arxiv_chunks.py.

Gold (R12/R13 of sparsevec-lexsem-graph: >=200 queries, STRICT truth):
  query = ONE random body sentence of a sampled chunk (8-40 words, mostly letters,
          no table rows); truth = that ONE chunk. n_relevant = 1, so recall@k is
          exactly hit@k. Reference chunks are excluded from corpus and queries.
Corpus: a seeded 20,000-chunk subset of the non-reference chunks, the same for every
arm. Scoring: exact cosine scan, no index -- this measures the REPRESENTATION only.

The window line is a mechanism check, not a result: it counts the queries whose
sentence ends beyond a model's token window, i.e. the model never saw it.

Run:  python src\\diag_dense_arms.py --arms m2v256,m2v128,minilm
      python src\\diag_dense_arms.py --arms nomic
"""
from __future__ import annotations

import argparse
import inspect
import os
import pickle
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

CACHE = ".tmp/arxiv_section_chunks.pkl"
N_CORPUS, N_Q, SEED = 20_000, 200, 0
M2V_256 = os.path.expanduser("~/models/m2v-minilm-l6-256")
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
NOMIC = "nomic-ai/nomic-embed-text-v1.5"
JINA = "jinaai/jina-embeddings-v5-text-nano"
JINA_M2V = os.path.expanduser("~/models/m2v-jina-v5-nano-256")
JINA_ROWS = 128256                                      # the tokenizer's full id range (model2vec keeps 128,001 of them)
SENT = re.compile(r"(?<=[.!?])\s+")


def corpus() -> list[dict]:
    recs = [r for r in pickle.load(open(CACHE, "rb"))["records"] if not r["is_reference"]]
    pick = np.sort(np.random.default_rng(SEED).choice(len(recs), N_CORPUS, replace=False))
    return [recs[i] for i in pick]


def pick_sentence(text: str, rng) -> tuple[str, int] | None:
    """Guarantee: (sentence, char offset of its end in `text`) or None."""
    body = "\n".join(l for l in text.split("\n")[1:] if not l.lstrip().startswith("#") and "|" not in l)
    cand = []
    for para in re.split(r"\n\s*\n", body):
        for s in SENT.split(para.replace("\n", " ")):
            w = s.split()
            if 8 <= len(w) <= 40 and sum(c.isalpha() or c == " " for c in s) / len(s) >= 0.85:
                cand.append(s.strip())
    if not cand:
        return None
    s = cand[rng.integers(len(cand))]
    head = " ".join(s.split()[:4])
    at = text.find(head)
    return s, (at + len(s)) if at >= 0 else -1


def queries(C: list[dict]):
    rng = np.random.default_rng(SEED + 1)
    out = []
    for i in rng.permutation(len(C)):
        got = pick_sentence(C[i]["text"], rng)
        if got:
            out.append((int(i), got[0], got[1]))
        if len(out) == N_Q:
            break
    assert len(out) == N_Q
    return out


def norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def enc_m2v(model, texts, multiprocessing=True):
    """Guarantee: L2-normalised static embeddings. multiprocessing=False for the 128k-row jina table: loky workers each copy it
    and Windows ran out of paging file (WinError 1455)."""
    params = inspect.signature(model.encode).parameters
    kw = {"max_length": None} if "max_length" in params else {}
    if "use_multiprocessing" in params and not multiprocessing:
        kw["use_multiprocessing"] = False
    return norm(model.encode(texts, show_progress_bar=False, **kw))


def enc_st(name, doc_texts, q_texts, max_len, doc_prefix="", q_prefix="", trust=False):
    import torch
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(name, device="cuda", trust_remote_code=trust)
    m.max_seq_length = max_len
    m.half()
    kw = dict(batch_size=32, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False)
    D = m.encode([doc_prefix + t for t in doc_texts], **kw)
    Q = m.encode([q_prefix + t for t in q_texts], **kw)
    del m
    torch.cuda.empty_cache()
    return D.astype(np.float32), Q.astype(np.float32)


def enc_windows(name, doc_texts, q_texts, win=250, cap=32):
    """Guarantee: (D, Q) where a document is the L2-normalised mean of its 256-token-window embeddings (at most `cap` windows, cut by
    the tokenizer's own character offsets), so the whole text is seen; queries are encoded whole."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    pieces, owner = [], []
    for i, t in enumerate(doc_texts):
        off = tok(t, add_special_tokens=False, return_offsets_mapping=True, truncation=False)["offset_mapping"]
        for k in range(0, min(len(off), win * cap), win):
            pieces.append(t[off[k][0]:off[min(k + win, len(off)) - 1][1]])
            owner.append(i)
    P, Qe = enc_st(name, pieces, q_texts, 256)
    D = np.zeros((len(doc_texts), P.shape[1]), np.float32)
    np.add.at(D, np.array(owner), norm(P))
    return D, Qe


def enc_jina(doc_texts, q_texts, max_len=8192, batch=8):
    """Guarantee: (D, Q) from jina-embeddings-v5-text-nano, retrieval task, query/document prompts, fp16 (the GPU is Turing: no bf16),
    sdpa attention (flash-attn is not installed), a window of max_len tokens."""
    import torch
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(JINA, trust_remote_code=True, device="cuda", model_kwargs={"dtype": torch.float16})
    m.max_seq_length = max_len
    kw = dict(batch_size=batch, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False, task="retrieval")
    D = m.encode(doc_texts, prompt_name="document", **kw)
    Qe = m.encode(q_texts, prompt_name="query", **kw)
    del m
    torch.cuda.empty_cache()
    return D.astype(np.float32), Qe.astype(np.float32)


def mrl(x: np.ndarray, dim: int) -> np.ndarray:
    """nomic v1.5 Matryoshka recipe: layer-norm, truncate, re-normalise."""
    x = x - x.mean(1, keepdims=True)
    x = x / np.sqrt((x ** 2).mean(1, keepdims=True) + 1e-5)
    return norm(x[:, :dim])


def score(D: np.ndarray, Q: np.ndarray, truth: list[int]) -> np.ndarray:
    """Guarantee: 0-based rank of each query's true chunk under exact cosine."""
    S = Q @ D.T
    return (S > S[np.arange(len(truth)), truth][:, None]).sum(1)


def report(name, dim, rank, secs, base=None, base_name="m2v-128"):
    boot = np.random.default_rng(0)
    hit = lambda k: (rank < k).astype(float)
    ci = lambda v: np.percentile([v[boot.integers(0, len(v), len(v))].mean() for _ in range(2000)], [2.5, 97.5])
    h10, h50 = hit(10), hit(50)
    lo, hi = ci(h50)
    line = "  %-22s %4d  rec@10 %.3f   rec@50 %.3f [%.3f, %.3f]  %5.0fs" % (name, dim, h10.mean(), h50.mean(), lo, hi, secs)
    if base is not None:
        d = h50 - (base < 50).astype(float)
        lo2, hi2 = np.percentile([d[boot.integers(0, len(d), len(d))].mean() for _ in range(2000)], [2.5, 97.5])
        line += "   vs %s %+.3f [%+.3f, %+.3f]" % (base_name, d.mean(), lo2, hi2)
    print(line, flush=True)


def window_check(C, Q):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MINILM)
    n = np.array([len(x) for x in tok([c["text"] for c in C], add_special_tokens=False)["input_ids"]])
    print("chunk tokens (MiniLM tokenizer): p50=%d p95=%d p99=%d max=%d" % tuple(np.percentile(n, [50, 95, 99, 100])))
    ends = [(len(tok(C[i]["text"][:e], add_special_tokens=False)["input_ids"]), e) for i, _, e in Q if e >= 0]
    pos = np.array([t for t, _ in ends])
    print("queries whose sentence ends beyond the window: 256 tok %d/%d   512 tok %d/%d"
          % ((pos > 256).sum(), len(pos), (pos > 512).sum(), len(pos)))


SECTION_ARMS = {"sec256": (256, 1e-4), "sec512": (512, 1e-4), "sec768": (None, 1e-4), "sec768nosif": (None, None)}   # arm -> (PCA dims, SIF coefficient)


def section_corpus() -> list[dict]:
    """Guarantee: a seeded N_CORPUS sample of the usable sections (src/section_corpus.py), in corpus order."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import section_corpus as sc
    recs = sc.load()
    pick = np.sort(np.random.default_rng(SEED).choice(len(recs), min(N_CORPUS, len(recs)), replace=False))
    return [recs[i] for i in pick]


def jina_static(tag: str, pca, sif):
    """Guarantee: the jina model distilled through model2vec with `pca` dimensions (None keeps all 768) and SIF coefficient `sif`
    (None: no frequency weighting), loaded from ~/models/m2v-jina-v5-nano-<tag> when it was saved, else distilled and saved."""
    from model2vec import StaticModel
    from model2vec.distill import distill
    path = os.path.expanduser("~/models/m2v-jina-v5-nano-" + tag)
    if os.path.isdir(path):
        return StaticModel.from_pretrained(path)
    m = distill(JINA, pca_dims=pca, sif_coefficient=sif, trust_remote_code=True)
    m.save_pretrained(path)
    return m


def section_arms(arms: list[str]) -> None:
    """Task T151. Sentence-to-section battery: query = one body sentence of a sampled section, truth = that section, exact cosine over a
    seeded 20,000-section corpus, section_embed.pool (no truncation). Every arm is jina through model2vec; the first arm run is the paired base."""
    import section_embed as se
    C = section_corpus()
    Q = queries(C)
    truth = [i for i, _, _ in Q]
    texts, qtexts = [c["text"] for c in C], [s for _, s, _ in Q]
    print("corpus %d sections (usable)  queries %d  n_relevant=1 each" % (len(C), len(Q)), flush=True)
    print("  %-26s %4s  %s" % ("arm", "dim", "recall (exact cosine scan)"), flush=True)
    base, base_name = None, ""
    for a in arms:
        pca, sif = SECTION_ARMS[a]
        t0 = time.time()
        m = jina_static(a.removeprefix("sec"), pca, sif)
        r = score(norm(se.pool(m, texts)), norm(se.pool(m, qtexts)), truth)
        name = "jina-model2vec %s (pca %s, sif %s)" % (a, pca or "none", sif)
        report(name, int(m.embedding.shape[1]), r, time.time() - t0, base, base_name)
        if base is None:
            base, base_name = r, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="m2v256,m2v128,minilm")
    ap.add_argument("--unit", default="chunk", choices=["chunk", "section"])
    args = ap.parse_args()
    arms = args.arms.split(",")
    if args.unit == "section":
        return section_arms(arms)
    C = corpus()
    Q = queries(C)
    truth = [i for i, _, _ in Q]
    texts, qtexts = [c["text"] for c in C], [s for _, s, _ in Q]
    print("corpus %d chunks (non-reference)  queries %d  n_relevant=1 each" % (len(C), len(Q)))
    window_check(C, Q)
    print("  %-22s %4s  %s" % ("arm", "dim", "recall (exact cosine scan)"))
    from model2vec.distill import distill
    base, base_name = None, "m2v-128"
    if "m2v128" in arms or "m2v256" in arms:            # the 2026-10-02 paired base; skipped unless asked (it re-encodes all 20,000 chunks on the CPU)
        t0 = time.time()
        m = distill(MINILM, pca_dims=128)
        base = score(enc_m2v(m, texts), enc_m2v(m, qtexts), truth)
    if "m2v128" in arms:
        report("model2vec MiniLM-128", 128, base, time.time() - t0)
    if "m2v256" in arms:
        from model2vec import StaticModel
        t0 = time.time()
        m = StaticModel.from_pretrained(M2V_256)
        r = score(enc_m2v(m, texts), enc_m2v(m, qtexts), truth)
        report("model2vec MiniLM-256", 256, r, time.time() - t0, base)
    if "minilm" in arms:
        t0 = time.time()
        D, Qe = enc_st(MINILM, texts, qtexts, 256)
        r = score(norm(D), norm(Qe), truth)
        report("MiniLM-L6 (256 tok)", 384, r, time.time() - t0, base, base_name)
        if base is None:                                # no m2v-128 base this run: every later arm is paired against MiniLM-256, the map's setting
            base, base_name = r, "MiniLM-256"
    if "minilm512" in arms:                             # the architecture's 512 positions, 256 is only what the code sets
        t0 = time.time()
        D, Qe = enc_st(MINILM, texts, qtexts, 512)
        report("MiniLM-L6 (512 tok)", 384, score(norm(D), norm(Qe), truth), time.time() - t0, base, base_name)
    if "minilmwin" in arms:
        t0 = time.time()
        D, Qe = enc_windows(MINILM, texts, qtexts)
        report("MiniLM-L6 windows pooled", 384, score(norm(D), norm(Qe), truth), time.time() - t0, base, base_name)
    if "jinam2v" in arms:                               # the operator's proposal: jina through model2vec (static, no window limit)
        from model2vec import StaticModel
        t0 = time.time()
        if os.path.isdir(JINA_M2V):
            m = StaticModel.from_pretrained(JINA_M2V)
        else:
            m = distill(JINA, pca_dims=256, trust_remote_code=True)
            m.save_pretrained(JINA_M2V)
        # A batch of texts is padded with the tokenizer's pad id (128004), which model2vec's 128,001 kept rows do not hold: IndexError in encode.
        # Zero rows and zero weights for the missing ids add nothing to a sum, so after L2 normalisation the cosine is unchanged.
        m.embedding = np.vstack([m.embedding, np.zeros((JINA_ROWS - m.embedding.shape[0], m.embedding.shape[1]), m.embedding.dtype)])
        m.weights = np.concatenate([m.weights, np.zeros(JINA_ROWS - len(m.weights), m.weights.dtype)])
        report("jina-v5-nano via model2vec-256", 256, score(enc_m2v(m, texts, False), enc_m2v(m, qtexts, False), truth), time.time() - t0, base, base_name)
    if "jina" in arms:
        t0 = time.time()
        D, Qe = enc_jina(texts, qtexts)
        report("jina-v5-nano (8192 tok)", int(D.shape[1]), score(norm(D), norm(Qe), truth), time.time() - t0, base, base_name)
    if "nomic" in arms:
        t0 = time.time()
        D, Qe = enc_st(NOMIC, texts, qtexts, 512, "search_document: ", "search_query: ", trust=True)
        el = time.time() - t0
        report("nomic-v1.5 (512 tok)", 768, score(norm(D), norm(Qe), truth), el, base)
        report("nomic-v1.5 MRL-128", 128, score(mrl(D, 128), mrl(Qe, 128), truth), el, base)


if __name__ == "__main__":
    main()
