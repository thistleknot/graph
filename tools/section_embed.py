"""section_embed.py -- a model2vec model applied to many texts, pooled the way model2vec pools but vectorised.

Spec: operator 2026-10-05 ("a model2vec version of that jina model applied to sections"; approved plan step A2). Task: playbook.md T150.
No other governing spec.

A static model embeds a text as the mean of its tokens' vectors (each times the token's weight), then L2-normalises. model2vec does that in a
Python loop per text, and its `encode` (a) truncates at 512 tokens unless told not to, (b) fails on a batch when the tokenizer pads with an id
past the rows it kept (jina's pad id 128004 against 128,001 rows; 2026-10-05), and (c) in multiprocessing mode copied the table into every
worker until Windows ran out of paging file. `pool` has none of the three: no truncation, ids without a row contribute nothing, one process,
and the sum over tokens is one `np.add.reduceat` per batch.

Guarantee of pool: out[i] equals StaticModel.encode([texts[i]], max_length=None) on every text whose ids all have rows, to float32 rounding
(cosine 1.0); a text with no tokens is the zero vector.
"""
from __future__ import annotations

import itertools

import numpy as np


BLOCK = 262_144                  # tokens gathered at a time: one 16.7M-character section is 25M tokens, 24 GB if gathered whole (2026-10-05)


def pool(model, texts: list[str], step: int = 500) -> np.ndarray:
    """Require: `model` a model2vec StaticModel (embedding, weights, normalize, tokenize). Guarantee: (len(texts), dim) float32."""
    emb, w = model.embedding, model.weights
    rows, dim = emb.shape
    model.tokenizer.no_padding()                # jina's tokenizer pads every text of a batch to the longest one with id 128004 (past the kept
    model.tokenizer.no_truncation()             # rows): 2,000 sections of 5.0M characters came out as 28.9M tokens. No pads, no cut.
    out = np.zeros((len(texts), dim), np.float32)
    for s in range(0, len(texts), step):
        ids = model.tokenize(texts[s:s + step], max_length=None)            # None: model2vec's encode would cut at 512 tokens
        lens = np.fromiter((len(x) for x in ids), np.int64, len(ids))
        total = int(lens.sum())
        if total == 0:
            continue
        flat = np.fromiter(itertools.chain.from_iterable(ids), np.int64, total)
        tid = np.repeat(np.arange(len(ids)), lens)                          # which text each token belongs to
        acc = np.zeros((len(ids), dim), np.float64)
        for a in range(0, total, BLOCK):                                    # a text may span blocks: its partial sums add into acc
            f = flat[a:a + BLOCK]
            keep = f < rows                                                 # an id with no row (jina's pad ids) adds nothing
            v = np.zeros((f.size, dim), np.float32)
            v[keep] = emb[f[keep]]
            if w is not None:
                v[keep] *= w[f[keep]][:, None]
            seg = tid[a:a + BLOCK]
            starts = np.concatenate([[0], np.flatnonzero(np.diff(seg)) + 1])
            acc[seg[starts]] += np.add.reduceat(v, starts, axis=0)
        nz = lens > 0
        out[s:s + len(ids)][nz] = (acc[nz] / lens[nz][:, None]).astype(np.float32)
    if model.normalize:
        out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-32
    return out


def center(E: np.ndarray, mu: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Guarantee: (rows minus the SAMPLE mean vector, then L2-normalised; the mean used). The inner product of two such rows is the cosine of
    their deviations from the corpus mean (repo guard R9's "correlation"): it removes the component every embedding shares, which otherwise makes
    all cosines near 1. It is NOT np.corrcoef between rows, which removes each row's own mean over dimensions; a zero row stays zero."""
    mu = E.mean(0) if mu is None else mu
    X = E - mu
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return (X / np.maximum(n, 1e-12)).astype(np.float32), mu
